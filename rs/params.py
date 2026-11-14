"""Parameter bundle with a content hash used as the export version tag.

All quantities are SI on the pose clock: seconds, radians, meters, pixels.
`t_ref` is mandatory: it is the reference exposure instant the corrected image
is aligned to. Files without it are rejected — there is no safe default,
because t_ref chooses which instant of the motion the output frame represents.
"""
import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class Params:
    camera: dict   # {fx_px, fy_px, cx_px, cy_px, width_px, height_px}
    readout: dict  # {readout_time_s, direction, t0_s, t_offset_s}
    t_ref: float   # seconds on the pose clock
    scene: dict    # {model: "rotation"} or {model: "plane", normal, dist_m, center0_m, velocity_m_s}
    motion: dict   # description of the pose track, e.g. {type: body_rate, omega_rad_s: [...], gyro_hz}
    notes: str = ""

    def to_dict(self):
        return {
            "camera": self.camera,
            "readout": self.readout,
            "t_ref_s": self.t_ref,
            "scene": self.scene,
            "motion": self.motion,
            "notes": self.notes,
            "units": {
                "time": "s",
                "angle": "rad",
                "length": "m",
                "focal": "px",
                "quat_order": "xyzw",
                "quat_meaning": "world->camera rotation",
            },
        }

    def canonical_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def version(self):
        return hashlib.sha1(self.canonical_json().encode("utf-8")).hexdigest()[:12]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d.pop("units", None)
        t_ref = d.get("t_ref_s", d.get("t_ref"))  # "t_ref" accepted for pre-t_ref_s files
        if t_ref is None:
            raise ValueError(
                "参数文件缺少时间参考 t_ref_s（参考曝光时刻，姿态时钟秒）。"
                "校正器不会默默猜测该值：请显式写入，例如取曝光窗口中点 "
                "t0_s + t_offset_s + readout_time_s/2。")
        return cls(
            camera=d["camera"],
            readout=d["readout"],
            t_ref=float(t_ref),
            scene=d["scene"],
            motion=d["motion"],
            notes=d.get("notes", ""),
        )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
