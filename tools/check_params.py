#!/usr/bin/env python3
"""检查参数组合与导出资源引用（维护工具）。

  python3 tools/check_params.py params.json [更多文件...]   # 校验参数文件
  python3 tools/check_params.py params.json --poses poses.json
  python3 tools/check_params.py --manifest out/manifest.json  # 校验整套导出

校验项（与 README「限制与程序行为」一一对应）：
  * 必填字段与单位：camera/readout/t_ref_s/scene/motion，角度一律弧度
  * t_ref_s 缺失 -> 报错（不猜测），并给出曝光窗口中点公式
  * readout.direction 必须是 rs.model.DIRECTIONS 之一（含倒序读出）
  * 给出姿态文件时：曝光窗口覆盖、采样密度（adequacy）、t_ref 落在轨迹跨度内
  * --manifest：清单中每个文件存在、每个 *_params_*.json 可加载、
    参数内容哈希与文件名中的版本标签一致（发现已失效参数/资源引用）

退出码：0 全部通过；1 存在问题。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs.model import DIRECTIONS
from rs.params import Params
from rs.poses import PoseTrack

SCENE_PLANE_KEYS = ("normal", "dist_m", "center0_m", "velocity_m_s")


def _is_vec(x, n=3):
    return isinstance(x, (list, tuple)) and len(x) == n and all(
        isinstance(v, (int, float)) for v in x)


def check_params_dict(d, path, problems):
    """逐项核对参数字典；problems 追加 (path, message)。"""
    def err(msg):
        problems.append((path, msg))

    cam = d.get("camera")
    if not isinstance(cam, dict):
        err("缺少 camera 段（fx_px/fy_px/cx_px/cy_px/width_px/height_px）")
    else:
        for k in ("fx_px", "fy_px", "cx_px", "cy_px"):
            if not isinstance(cam.get(k), (int, float)):
                err(f"camera.{k} 缺失或不是数值（单位 px）")
        for k in ("width_px", "height_px"):
            if not isinstance(cam.get(k), int) or cam.get(k, 0) <= 0:
                err(f"camera.{k} 缺失或不是正整数")
        if isinstance(cam.get("fx_px"), (int, float)) and cam["fx_px"] <= 0:
            err("camera.fx_px 必须为正")

    rd = d.get("readout")
    if not isinstance(rd, dict):
        err("缺少 readout 段（readout_time_s/direction/t0_s/t_offset_s）")
    else:
        rt = rd.get("readout_time_s")
        if not isinstance(rt, (int, float)) or rt <= 0:
            err("readout.readout_time_s 必须为正数（首行至末行的总时长，秒）")
        if rd.get("direction") not in DIRECTIONS:
            err(f"readout.direction 必须是 {DIRECTIONS} 之一；"
                "注意行号 0 的位置随方向不同（见 README 行时序约定）")
        for k in ("t0_s", "t_offset_s"):
            if k in rd and not isinstance(rd[k], (int, float)):
                err(f"readout.{k} 不是数值（秒）")

    if "t_ref_s" not in d and "t_ref" not in d:
        err("缺少 t_ref_s（参考曝光时刻，姿态时钟秒）。不会默默猜测："
            "请显式写入，例如 t0_s + t_offset_s + readout_time_s/2")
    elif not isinstance(d.get("t_ref_s", d.get("t_ref")), (int, float)):
        err("t_ref_s 不是数值（秒）")

    sc = d.get("scene")
    if not isinstance(sc, dict) or sc.get("model") not in ("rotation", "plane"):
        err('scene.model 必须是 "rotation" 或 "plane"')
    elif sc["model"] == "plane":
        for k in SCENE_PLANE_KEYS:
            if k not in sc:
                err(f"scene.{k} 缺失（平面模型需要 {SCENE_PLANE_KEYS}，"
                    "旧键名 dist/center0/velocity 已失效）")
        if _is_vec(sc.get("normal")) and np.linalg.norm(sc["normal"]) < 1e-12:
            err("scene.normal 不能是零向量")
        if isinstance(sc.get("dist_m"), (int, float)) and sc["dist_m"] <= 0:
            err("scene.dist_m 必须为正（米）")
        for k in ("center0_m", "velocity_m_s"):
            if k in sc and not _is_vec(sc[k]):
                err(f"scene.{k} 必须是三分量数组（米 / 米每秒）")

    mo = d.get("motion")
    if not isinstance(mo, dict) or "type" not in mo:
        err("缺少 motion 段（至少含 type；仅作记录，校正读取的是姿态文件）")
    elif mo.get("type") == "body_rate":
        w = mo.get("omega_rad_s")
        if not _is_vec(w):
            err("motion.omega_rad_s 必须是三分量数组（弧度/秒，机体/相机系）")
        elif np.linalg.norm(w) > 50.0:
            err("motion.omega_rad_s 模长超过 50 rad/s，疑似把度/秒当成了弧度/秒")

    units = d.get("units")
    if units is not None and units.get("angle") not in (None, "rad"):
        err(f"units.angle 声明为 {units.get('angle')!r}，校正器只接受弧度")

    try:  # 最终由真实读入代码复核一遍
        Params.from_dict(d)
    except (ValueError, KeyError, TypeError) as e:
        err(f"Params 读入失败：{e}")


def check_with_poses(d, poses_path, problems):
    try:
        params = Params.from_dict(d)
    except (ValueError, KeyError, TypeError):
        return  # 已由 check_params_dict 报告
    with open(poses_path, "r", encoding="utf-8") as f:
        pj = json.load(f)
    if "quats_xyzw" in pj and "times_s" in pj:
        track = PoseTrack(np.asarray(pj["times_s"], float),
                          np.asarray(pj["quats_xyzw"], float))
    elif "gyro_times_s" in pj and "gyro_rad_s" in pj:
        track = PoseTrack.integrate_gyro(np.asarray(pj["gyro_times_s"], float),
                                         np.asarray(pj["gyro_rad_s"], float),
                                         pj.get("q0_xyzw", (0, 0, 0, 1)))
    else:
        problems.append((poses_path, "姿态文件缺少 times_s/quats_xyzw 或 "
                                     "gyro_times_s/gyro_rad_s（时间基准必须显式给出）"))
        return
    rd = params.readout
    t_lo = rd.get("t0_s", 0.0) + rd.get("t_offset_s", 0.0)
    rep = track.adequacy(t_lo, t_lo + rd["readout_time_s"])
    for p in rep["problems"]:
        problems.append((poses_path, f"姿态轨迹不足：{p['type']} — {p['msg']}"))
    if not (track.times[0] <= params.t_ref <= track.times[-1]):
        problems.append((poses_path,
                         f"t_ref_s={params.t_ref} 不在姿态跨度 "
                         f"[{track.times[0]}, {track.times[-1]}] 内"))


def check_manifest(path, problems):
    base = os.path.dirname(os.path.abspath(path))
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    seen = set()
    for m in entries:
        for fn in m["files"]:
            full = os.path.join(base, fn)
            if not os.path.isfile(full):
                problems.append((path, f"清单引用的文件不存在：{fn}"))
                continue
            seen.add(os.path.abspath(full))
            if fn.endswith("_params.json") or "_params_" in fn:
                try:
                    with open(full, "r", encoding="utf-8") as pf:
                        d = json.load(pf)
                    ver = Params.from_dict(d).version()
                    tag = fn.rsplit("_", 1)[-1].split(".")[0]
                    if ver != tag:
                        problems.append((full, f"参数内容哈希 {ver} 与文件名版本标签 "
                                               f"{tag} 不一致（文件已失效，请重新导出）"))
                    check_params_dict(d, full, problems)
                except (ValueError, KeyError, json.JSONDecodeError) as e:
                    problems.append((full, f"参数文件无法加载：{e}"))
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", nargs="*", help="参数 JSON 文件（支持通配符）")
    ap.add_argument("--poses", help="可选姿态 JSON，联合检查时间覆盖")
    ap.add_argument("--manifest", help="导出清单 manifest.json，检查资源引用与版本标签")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出问题列表")
    a = ap.parse_args()

    problems = []
    files = [p for pat in a.params for p in (glob.glob(pat) or [pat])]
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            problems.append((path, f"无法读取：{e}"))
            continue
        check_params_dict(d, path, problems)
        if a.poses:
            check_with_poses(d, a.poses, problems)
    if a.manifest:
        check_manifest(a.manifest, problems)
    if not files and not a.manifest:
        ap.error("没有要检查的对象（给出参数文件或 --manifest）")

    if a.json:
        print(json.dumps([{"file": f, "problem": m} for f, m in problems],
                         ensure_ascii=False, indent=2))
    else:
        if problems:
            for f, m in problems:
                print(f"[FAIL] {f}: {m}")
        else:
            print("全部通过：参数组合与资源引用一致")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
