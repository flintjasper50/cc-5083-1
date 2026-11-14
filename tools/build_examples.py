#!/usr/bin/env python3
"""可执行核验入口：在干净目录中重建全套示例，并从导出结果生成文档数据。

  python3 tools/build_examples.py                 # 重建 out/ 全套示例 + docs/ 图表
  python3 tools/build_examples.py --only grid_mixed --width 320 --height 240

行为：
  1. 目标目录整体清空重建（不带旧缓存），按 EXAMPLES 矩阵逐例运行 pipeline；
  2. 每例经 Exporter 导出图像/映射/参数（版本标签 = 参数内容哈希）；
  3. 控制点误差由本脚本重新读入导出的 .npz 映射计算（不用内存中的中间量）；
  4. 示意图（输入/校正/有效性、行时序图）从真实导出结果写入 docs/figures/；
  5. 指标表注入 README.md 的 <!-- METRICS:BEGIN/END --> 之间；
  6. 最后用 check_params 复核 manifest：已失效参数与资源引用会导致非零退出。
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs import Exporter, imaging, pipeline, synth
from rs.camera import Intrinsics
from rs.model import PlaneScene, ReadoutSpec, RotationScene
from rs.params import Params
from rs.poses import PoseTrack

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 全套示例矩阵：覆盖 README 写出的每种限制情形；expect 中的断言即
# 「说明写出的限制确实对应程序行为」的可执行核验，任一不满足则构建失败。
EXAMPLES = [
    {"name": "grid_mixed", "cfg": {"scene": "grid", "motion": "mixed"},
     "note": "默认情形：混合角速度，自上而下读出",
     "expect": [("姿态覆盖充分", lambda r, s: s["pose_report"]["ok"]),
                ("映射收敛", lambda r, s: s["converged"]),
                ("控制点均值 < 0.2 px", lambda r, s: r["cp_mean_px"] is not None
                 and r["cp_mean_px"] < 0.2)]},
    {"name": "grid_reversed", "cfg": {"scene": "grid", "motion": "mixed",
                                      "direction": "bottom_to_top"},
     "note": "倒序读出：行号 0 在图像底边",
     "expect": [("倒序读出同样收敛", lambda r, s: s["converged"]),
                ("控制点均值 < 0.3 px", lambda r, s: r["cp_mean_px"] is not None
                 and r["cp_mean_px"] < 0.3)]},
    {"name": "grid_static", "cfg": {"scene": "grid", "motion": "static"},
     "note": "零角速度：校正必须退化为恒等",
     "expect": [("控制点均值 < 0.05 px", lambda r, s: r["cp_mean_px"] is not None
                 and r["cp_mean_px"] < 0.05),
                ("平坦区 MAE < 0.01", lambda r, s: s["mae_flat"] < 0.01)]},
    {"name": "lines_yaw", "cfg": {"scene": "lines", "motion": "yaw"},
     "note": "竖线场景：偏航导致的倾斜应被拉直",
     "expect": [("校正后线摆动 < 0.35 px", lambda r, s:
                 s["line_wobble"] is not None
                 and s["line_wobble"]["corrected_px"] < 0.35)]},
    {"name": "plane_mixed", "cfg": {"scene": "plane", "model": "plane", "motion": "mixed"},
     "note": "已知平面 + 平移：旋转模型不够，需平面模型",
     "expect": [("控制点均值 < 0.3 px", lambda r, s: r["cp_mean_px"] is not None
                 and r["cp_mean_px"] < 0.3)]},
    {"name": "grid_extreme", "cfg": {"scene": "grid", "motion": "extreme",
                                     "readout_ms": 60.0, "gyro_hz": 1000.0},
     "note": "剧烈运动：映射折叠与越界区域必须被标记",
     "expect": [("折叠区域被标记", lambda r, s: s["reason_counts"]["folding"] > 0),
                ("越界区域被标记", lambda r, s: s["reason_counts"]["out_of_bounds"] > 0)]},
    {"name": "grid_sparse", "cfg": {"scene": "grid", "gyro_hz": 8.0},
     "note": "采样过疏（相邻姿态转角超限）：姿态报告必须拒绝",
     "expect": [("姿态报告拒绝", lambda r, s: not s["pose_report"]["ok"]),
                ("报告含 sparse", lambda r, s: "sparse" in
                 {p["type"] for p in s["pose_report"]["problems"]})]},
]


def control_point_errors_from_export(map_npz_path, control_points):
    """只读导出的 .npz，对每个参考控制点做映射闭环检查。

    合成导出的 .npz 内含真值前向映射 fmap_uv：对参考点 (x,y) 取导出逆映射
    给出的输入坐标 q，再用 fmap_uv 把 q 映回参考系，应恰好回到 (x,y)。
    若导出不含 fmap_uv（真实照片），退化为投影一致性检查：由 q 的行号推出
    曝光时刻 t，用导出的姿态轨迹把 (x,y) 的参考光线投影回输入图像。
    返回每项误差（px）。映射无效的点跳过。
    """
    z = np.load(map_npz_path, allow_pickle=False)
    map_uv = z["map_uv"]
    reasons = z["reasons"]
    H, W = map_uv.shape[:2]

    if "fmap_uv" in z:
        from rs.corrector import bilinear_sample
        fmap = z["fmap_uv"]
        fvalid = z["fmap_valid"] if "fmap_valid" in z else np.ones((H, W), bool)
        errs = []
        for (x, y) in control_points:
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < W and 0 <= yi < H) or reasons[yi, xi] != 0:
                continue
            if not fvalid[yi, xi]:
                continue
            back = bilinear_sample(fmap, map_uv[yi, xi][None, :])[0]
            errs.append(float(np.linalg.norm(back - np.array([x, y]))))
        return errs

    params = Params.from_dict(json.loads(str(z["params_json"])))
    track = PoseTrack(z["pose_times_s"], z["pose_quats_xyzw"])
    K = Intrinsics(*[float(v) for v in z["K"][:4]])
    cam = params.camera
    rd = params.readout
    readout = ReadoutSpec(cam["width_px"], cam["height_px"], rd["readout_time_s"],
                          rd["direction"], rd["t0_s"], rd["t_offset_s"])
    t_ref = float(z["t_ref_s"])
    q_ref, _ = track.interp(np.array(t_ref))
    sc = params.scene
    if sc["model"] == "plane":
        scene = PlaneScene(track, q_ref, K, sc["normal"], sc["dist_m"],
                           sc["center0_m"], sc["velocity_m_s"], t_ref)
    else:
        scene = RotationScene(track, q_ref, K)

    errs = []
    for (x, y) in control_points:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H) or reasons[yi, xi] != 0:
            continue
        q = map_uv[yi, xi]
        t = readout.line_time(readout.readout_index(q[0], q[1]))
        ray = np.array([[(x - K.cx) / K.fx, (y - K.cy) / K.fy, 1.0]])
        uv_back, zz, ok = scene.to_input(ray, np.array([t]))
        if ok[0] and zz[0] > 1e-9:
            errs.append(float(np.linalg.norm(uv_back[0] - q)))
    return errs


def scene_control_points(cfg, W, H):
    if cfg["scene"] == "lines":
        _, bars = synth.scene_lines(W, H)
        return [(float(x), float(y)) for x in bars for y in range(60, H, 120)]
    _, pts = synth.scene_grid(W, H)
    return pts


def draw_row_timing(map_npz_path, out_png):
    """行时序示意图：用导出映射的 row_times_s 按图像行位置绘制（真实结果）。

    横轴是图像行 v（横向读出时为列 u），因此倒序读出呈现为下降曲线。
    """
    from PIL import Image, ImageDraw, ImageFont
    z = np.load(map_npz_path, allow_pickle=False)
    rt = np.asarray(z["row_times_s"], dtype=np.float64) * 1000.0
    params = json.loads(str(z["params_json"]))
    direction = params["readout"]["direction"]
    n = rt.size
    if direction == "top_to_bottom":
        pos = np.arange(n, dtype=np.float64)
        x_label = f"image row v (0 .. {n - 1})"
    elif direction == "bottom_to_top":
        pos = (n - 1) - np.arange(n, dtype=np.float64)
        x_label = f"image row v (0 .. {n - 1}), line 0 at bottom"
    elif direction == "left_to_right":
        pos = np.arange(n, dtype=np.float64)
        x_label = f"image col u (0 .. {n - 1})"
    else:
        pos = (n - 1) - np.arange(n, dtype=np.float64)
        x_label = f"image col u (0 .. {n - 1}), line 0 at right"
    order = np.argsort(pos)
    pos, rt = pos[order], rt[order]

    Wp, Hp, m = 560, 320, 46
    img = Image.new("RGB", (Wp, Hp), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    dr.rectangle([m, 12, Wp - 12, Hp - m], outline=(60, 60, 60))
    t0, t1 = float(rt.min()), float(rt.max())
    span = max(t1 - t0, 1e-9)
    pts = [(m + (Wp - 12 - m) * p / (n - 1),
            12 + (Hp - m - 12) * (1.0 - (t - t0) / span)) for p, t in zip(pos, rt)]
    dr.line(pts, fill=(11, 107, 203), width=2)
    dr.text((m, Hp - m + 8), x_label, fill=(30, 30, 30), font=font)
    dr.text((m, 2), f"exposure {t0:.2f} .. {t1:.2f} ms (first..last line)",
            fill=(30, 30, 30), font=font)
    dr.text((Wp - 130, 2), f"span {t1 - t0:.1f} ms", fill=(30, 30, 30), font=font)
    img.save(out_png)


def build(dir_out, docs_dir, readme_path, only=None, size=None):
    if os.path.isdir(dir_out):
        shutil.rmtree(dir_out)  # 干净目录：不带旧缓存
    os.makedirs(dir_out)
    fig_dir = os.path.join(docs_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rows, failures = [], []
    for ex in EXAMPLES:
        if only and ex["name"] not in only:
            continue
        cfg = dict(ex["cfg"])
        if size:
            cfg["width"], cfg["height"] = size
        res = pipeline.run_synthetic(cfg)
        W, H = res["cfg"]["width"], res["cfg"]["height"]
        exp = Exporter(dir_out)
        out = exp.export(
            name=ex["name"], params=res["params"],
            images={"reference": res["images"]["reference"],
                    "input": res["images"]["input"],
                    "corrected": res["images"]["corrected"],
                    "validity": imaging.validity_image(res["inv"].reasons).astype(float) / 255.0},
            inv=res["inv"], track=res["track_corr"], K=res["K"],
            extra_npz={"fmap_uv": res["fmap"], "fmap_valid": res["fvalid"]})
        ver = out["version"]
        map_npz = os.path.join(dir_out, f"{ex['name']}_map_{ver}.npz")

        cps = scene_control_points(res["cfg"], W, H)
        errs = control_point_errors_from_export(map_npz, cps)
        s = res["stats"]
        row = {
            "name": ex["name"], "note": ex["note"], "version": ver,
            "direction": res["cfg"]["direction"], "motion": res["cfg"]["motion"],
            "cp_n": len(errs),
            "cp_mean_px": float(np.mean(errs)) if errs else None,
            "cp_max_px": float(np.max(errs)) if errs else None,
            "mae_flat": s["mae_flat"], "valid_fraction": s["valid_fraction"],
            "converged": s["converged"], "pose_ok": s["pose_report"]["ok"],
            "reason_counts": s["reason_counts"],
        }
        rows.append(row)
        for desc, ok in ex.get("expect", []):
            if not ok(row, s):
                failures.append((ex["name"], f"期望不满足：{desc}"))

        for tag in ("input", "corrected", "validity"):
            shutil.copyfile(os.path.join(dir_out, f"{ex['name']}_{tag}_{ver}.png"),
                            os.path.join(fig_dir, f"{ex['name']}_{tag}.png"))
        draw_row_timing(map_npz, os.path.join(fig_dir, f"{ex['name']}_row_timing.png"))
        print(f"[{ex['name']}] 版本 {ver} 控制点均值 "
              f"{row['cp_mean_px'] if errs else float('nan'):.4f} px")

    metrics_path = os.path.join(docs_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if os.path.exists(readme_path):
        update_readme_metrics(readme_path, rows)
    else:
        print(f"警告：{readme_path} 不存在，跳过指标注入", file=sys.stderr)

    # 复核：已失效参数与资源引用
    spec = importlib.util.spec_from_file_location(
        "check_params", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "check_params.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    problems = []
    mod.check_manifest(os.path.join(dir_out, "manifest.json"), problems)
    for f, m in problems:
        failures.append((f, m))
    return rows, failures


METRICS_BEGIN = "<!-- METRICS:BEGIN -->"
METRICS_END = "<!-- METRICS:END -->"


def render_metrics_table(rows):
    def num(x):
        return f"{x:.4g}" if x is not None else "—"
    head = ["| 示例 | 说明 | 版本 | 控制点均值 (px) | 控制点最大 (px) | 平坦区 MAE | 有效比例 | 姿态检查 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    body = []
    for r in rows:
        body.append(f"| `{r['name']}` | {r['note']} | `{r['version']}` "
                    f"| {num(r['cp_mean_px'])} | {num(r['cp_max_px'])} "
                    f"| {num(r['mae_flat'])} | {r['valid_fraction']:.3f} "
                    f"| {'通过' if r['pose_ok'] else '拒绝'} |")
    return "\n".join([METRICS_BEGIN,
                      "（本表由 `tools/build_examples.py` 读取导出映射重新计算，请勿手改）",
                      *head, *body, METRICS_END])


def update_readme_metrics(readme_path, rows):
    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()
    if METRICS_BEGIN not in text or METRICS_END not in text:
        raise SystemExit(f"{readme_path} 缺少 {METRICS_BEGIN} / {METRICS_END} 标记，"
                         "无法注入指标表")
    pre = text.split(METRICS_BEGIN)[0]
    post = text.split(METRICS_END)[1]
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(pre + render_metrics_table(rows) + post)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(ROOT, "out"), help="示例输出目录（会被清空重建）")
    ap.add_argument("--docs", default=os.path.join(ROOT, "docs"))
    ap.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    ap.add_argument("--only", help="只跑指定示例（逗号分隔）")
    ap.add_argument("--width", type=int, help="覆盖图像宽度（快速自检用）")
    ap.add_argument("--height", type=int, help="覆盖图像高度（快速自检用）")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    size = (a.width, a.height) if a.width and a.height else None
    rows, failures = build(a.dir, a.docs, a.readme, only=only, size=size)
    print(f"\n完成 {len(rows)} 个示例；指标写入 docs/metrics.json 并注入 README")
    if failures:
        for f, m in failures:
            print(f"[FAIL] {f}: {m}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
