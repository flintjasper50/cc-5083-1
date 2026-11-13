"""Parameter bundle with a content hash used as the export version tag."""
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
        return cls(
            camera=d["camera"],
            readout=d["readout"],
            t_ref=d.get("t_ref_s", d.get("t_ref")),
            scene=d["scene"],
            motion=d["motion"],
            notes=d.get("notes", ""),
        )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
