"""Versioned export: corrected image, inverse map and parameters always share one version tag."""
import json
import os
from datetime import datetime, timezone

import numpy as np

from . import imaging


class Exporter:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def export(self, name, params, images, inv, track, K):
        """images: {tag: float HxWx3}. Returns dict with version and written file paths."""
        ver = params.version()
        files = []
        for tag, img in images.items():
            p = os.path.join(self.out_dir, f"{name}_{tag}_{ver}.png")
            imaging.save_png(p, img)
            files.append(p)

        map_path = os.path.join(self.out_dir, f"{name}_map_{ver}.npz")
        np.savez(
            map_path,
            map_uv=inv.uv.astype(np.float64),
            reasons=inv.reasons,
            row_times_s=inv.row_times,
            pose_times_s=track.times,
            pose_quats_xyzw=track.quats,
            K=np.array([K.fx, K.fy, K.cx, K.cy]),
            t_ref_s=np.float64(params.t_ref),
            params_json=params.canonical_json(),
        )
        files.append(map_path)

        params_path = os.path.join(self.out_dir, f"{name}_params_{ver}.json")
        params.save(params_path)
        files.append(params_path)

        manifest_path = os.path.join(self.out_dir, "manifest.json")
        manifest = []
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        manifest = [m for m in manifest if not (m["name"] == name and m["version"] == ver)]
        manifest.append({
            "name": name,
            "version": ver,
            "created": datetime.now(timezone.utc).isoformat(),
            "files": [os.path.basename(p) for p in files],
        })
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        files.append(manifest_path)
        return {"version": ver, "files": files}
