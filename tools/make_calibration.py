#!/usr/bin/env python3
"""生成标定图片与配套样例（参数、姿态、合成滚动快门照片）。

用法：
  python3 tools/make_calibration.py --out examples

产物（--out 目录）：
  calib_grid.png / calib_lines.png  打印或上屏用的标定图案（网格控制点 / 竖线）
  sample_reference.png              全局快门参考图（合成样例的真值）
  sample_capture.png                同一图案的滚动快门“拍摄”结果，可直接用于校正
  sample_params.json                与样例匹配的参数文件（含 t_ref_s，可被 demo.py photo 读入）
  sample_poses.json                 与样例匹配的姿态轨迹（显式时间戳 + 四元数）

样例文件同时是 check_params.py 与 demo.py photo 的冒烟输入，随仓库提交。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs import imaging, synth
from rs.camera import Intrinsics
from rs.model import ReadoutSpec
from rs.params import Params
from rs.poses import PoseTrack

# 样例拍摄设定：与 README「参数逐项对照」中的示例值一致
SAMPLE = {
    "width": 640, "height": 480, "fx": 560.0, "fy": 560.0,
    "readout_ms": 32.0, "direction": "top_to_bottom",
    "t_offset_ms": 2.0, "t_ref_ms": 16.0,
    "omega": (0.12, 0.5, 0.08),  # rad/s，机体（相机）系
}


def build_sample(out_dir):
    W, H = SAMPLE["width"], SAMPLE["height"]
    K = Intrinsics(SAMPLE["fx"], SAMPLE["fy"], (W - 1) / 2.0, (H - 1) / 2.0)
    readout_s = SAMPLE["readout_ms"] / 1000.0
    t_ref = SAMPLE["t_ref_ms"] / 1000.0
    t_offset = SAMPLE["t_offset_ms"] / 1000.0

    readout = ReadoutSpec(W, H, readout_s, SAMPLE["direction"], t0=0.0, t_offset=t_offset)
    t_lo, t_hi = readout.time_range()
    times = np.arange(t_lo - 0.01, t_hi + 0.01, 0.001)
    track = PoseTrack.from_body_rate(times, SAMPLE["omega"])
    q_ref, _ = track.interp(np.array(t_ref))

    ref_img, _ = synth.scene_grid(W, H)
    capture, _, _ = synth.forward_render(ref_img, K, readout, track, q_ref, t_ref)

    params = Params(
        camera={"fx_px": K.fx, "fy_px": K.fy, "cx_px": K.cx, "cy_px": K.cy,
                "width_px": W, "height_px": H},
        readout={"readout_time_s": readout_s, "direction": SAMPLE["direction"],
                 "t0_s": 0.0, "t_offset_s": t_offset},
        t_ref=t_ref,
        scene={"model": "rotation"},
        motion={"type": "body_rate", "omega_rad_s": list(SAMPLE["omega"]),
                "pose_span_s": [float(times[0]), float(times[-1])]},
        notes="make_calibration.py 生成的样例；拍摄时刻 = 姿态时钟",
    )
    poses = {"times_s": [round(float(t), 6) for t in track.times],
             "quats_xyzw": [[round(float(x), 10) for x in q] for q in track.quats]}

    written = []
    for name, img in [("calib_grid.png", synth.scene_grid(W, H)[0]),
                      ("calib_lines.png", synth.scene_lines(W, H)[0]),
                      ("sample_reference.png", ref_img),
                      ("sample_capture.png", capture)]:
        p = os.path.join(out_dir, name)
        imaging.save_png(p, img)
        written.append(p)
    p = os.path.join(out_dir, "sample_params.json")
    params.save(p)
    written.append(p)
    p = os.path.join(out_dir, "sample_poses.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(poses, f)
    written.append(p)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="examples", help="输出目录（默认 examples/）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in build_sample(a.out):
        print("写出", p)


if __name__ == "__main__":
    main()
