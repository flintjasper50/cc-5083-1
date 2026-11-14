#!/usr/bin/env python3
"""Rolling-shutter corrector demo / CLI.

  python3 demo.py synth  --scene grid --motion mixed ...   # synthetic verification
  python3 demo.py photo  --image in.png --params p.json --poses poses.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rs import (Exporter, Intrinsics, Params, PoseTrack, imaging, pipeline)


def cmd_synth(a):
    cfg = {
        "scene": a.scene, "direction": a.direction, "readout_ms": a.readout_ms,
        "t_offset_ms": a.offset_ms, "t_ref_ms": a.t_ref_ms, "gyro_hz": a.gyro_hz,
        "motion": a.motion, "model": a.model, "moving": a.moving,
    }
    res = pipeline.run_synthetic(cfg)
    s = res["stats"]
    print(json.dumps(s, indent=2, ensure_ascii=False))

    if a.sweep_offset:
        print("\n时间偏移扫描（校正侧 offset -> 有效区 MAE，真实偏移 %.1f ms）："
              % (pipeline.TRUE_OFFSET_S * 1000))
        for off in np.linspace(a.offset_ms - 3, a.offset_ms + 3, 7):
            r = pipeline.run_synthetic({**cfg, "t_offset_ms": float(off)})
            print("  %+5.1f ms -> MAE %.5f, 控制点均值 %.3f px"
                  % (off, r["stats"]["mae_valid"], r["stats"]["control_points"]["mean_px"]))

    exp = Exporter(a.out)
    out = exp.export(
        name=f"synth_{a.scene}",
        params=res["params"],
        images={
            "reference": res["images"]["reference"],
            "input": res["images"]["input"],
            "corrected": res["images"]["corrected"],
            "validity": imaging.validity_image(res["inv"].reasons).astype(np.float64) / 255.0,
        },
        inv=res["inv"], track=res["track_corr"], K=res["K"],
        extra_npz={"fmap_uv": res["fmap"], "fmap_valid": res["fvalid"]},
    )
    print("\n导出版本 %s:" % out["version"])
    for f in out["files"]:
        print("  " + f)


def _load_track(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    try:
        return PoseTrack.from_dict(d)
    except ValueError as e:
        raise SystemExit(f"姿态文件无效：{e}")


def cmd_photo(a):
    try:
        params = Params.load(a.params)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        raise SystemExit(f"参数文件无效：{e}")
    track = _load_track(a.poses)
    img = imaging.load_image(a.image)
    try:
        corrected, inv, stats = pipeline.run_photo(img, params, track)
    except ValueError as e:
        raise SystemExit(str(e))
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    cam = params.camera
    K = Intrinsics(cam["fx_px"], cam["fy_px"], cam["cx_px"], cam["cy_px"])
    exp = Exporter(a.out)
    out = exp.export(
        name="photo", params=params,
        images={"input": img, "corrected": corrected,
                "validity": imaging.validity_image(inv.reasons).astype(np.float64) / 255.0},
        inv=inv, track=track, K=K)
    print("导出版本 %s:" % out["version"])
    for f in out["files"]:
        print("  " + f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="合成场景端到端验证")
    s.add_argument("--scene", choices=["grid", "lines", "plane"], default="grid")
    s.add_argument("--direction", choices=["top_to_bottom", "bottom_to_top",
                                           "left_to_right", "right_to_left"],
                   default="top_to_bottom")
    s.add_argument("--readout-ms", type=float, default=32.0)
    s.add_argument("--offset-ms", type=float, default=pipeline.TRUE_OFFSET_S * 1000,
                   help="校正侧时间偏移（真实值 %.1f ms）" % (pipeline.TRUE_OFFSET_S * 1000))
    s.add_argument("--t-ref-ms", type=float, default=16.0)
    s.add_argument("--gyro-hz", type=float, default=500.0)
    s.add_argument("--motion", choices=list(pipeline.OMEGAS), default="mixed")
    s.add_argument("--model", choices=["rotation", "plane"], default="rotation")
    s.add_argument("--moving", action="store_true", help="加入独立运动物体（不支持，仅标记）")
    s.add_argument("--sweep-offset", action="store_true", help="扫描时间偏移展示敏感性")
    s.add_argument("--out", default="out")
    s.set_defaults(fn=cmd_synth)

    ph = sub.add_parser("photo", help="校正真实照片（需参数与姿态文件）")
    ph.add_argument("--image", required=True)
    ph.add_argument("--params", required=True, help="Params JSON（camera/readout/t_ref/scene）")
    ph.add_argument("--poses", required=True,
                    help='{"times_s":[...],"quats_xyzw":[[x,y,z,w],...]} 或 '
                         '{"gyro_times_s":[...],"gyro_rad_s":[[wx,wy,wz],...],"q0_xyzw":[...]}')
    ph.add_argument("--out", default="out")
    ph.set_defaults(fn=cmd_photo)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
