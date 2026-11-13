import json
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs import (Exporter, Intrinsics, Params, PlaneScene, PoseTrack, ReadoutSpec,
                RotationScene, apply_inverse_map, build_inverse_map, detect_folding, lie,
                pipeline, synth)
from rs.corrector import (REASON_BOUNDS, REASON_FOLD, REASON_MOVING, REASON_POSE)

FAST = {"width": 320, "height": 240, "gyro_hz": 400.0}


def run(cfg):
    return pipeline.run_synthetic({**FAST, **cfg})


class TestLie(unittest.TestCase):
    def test_rotvec_roundtrip(self):
        rng = np.random.default_rng(0)
        rv = rng.normal(size=(50, 3)) * 0.7
        rv2 = lie.quat_to_rotvec(lie.quat_from_rotvec(rv))
        self.assertTrue(np.allclose(rv, rv2, atol=1e-10))

    def test_rotate_matches_matrix(self):
        rng = np.random.default_rng(1)
        q = lie.quat_from_rotvec(rng.normal(size=3))
        v = rng.normal(size=(20, 3))
        self.assertTrue(np.allclose(lie.quat_rotate(q, v),
                                    (lie.quat_to_matrix(q) @ v.T).T, atol=1e-12))

    def test_slerp_midpoint(self):
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = lie.quat_from_rotvec(np.array([0.0, 0.0, 0.4]))
        qm = lie.slerp(q0, q1, np.array(0.5))
        ang = lie.quat_angle_between(q0, qm)
        self.assertAlmostEqual(float(ang), 0.2, places=10)


class TestPoses(unittest.TestCase):
    def test_gyro_integration_matches_analytic(self):
        t = np.linspace(0.0, 0.05, 501)
        omega = np.array([0.1, 0.5, 0.08])
        analytic = PoseTrack.from_body_rate(t, omega)
        gyro = PoseTrack.integrate_gyro(t, np.broadcast_to(omega, (t.size, 3)))
        err = lie.quat_angle_between(analytic.quats, gyro.quats)
        self.assertLess(float(np.max(err)), 1e-8)

    def test_adequacy_coverage_and_density(self):
        t = np.linspace(0.0, 0.01, 3)
        tr = PoseTrack.from_body_rate(t, (0.0, 30.0, 0.0))
        rep = tr.adequacy(-0.005, 0.02)
        kinds = {p["type"] for p in rep["problems"]}
        self.assertIn("coverage_start", kinds)
        self.assertIn("coverage_end", kinds)
        self.assertIn("sparse", kinds)
        self.assertFalse(rep["ok"])
        dense = PoseTrack.from_body_rate(np.linspace(0.0, 0.05, 501), (0.0, 0.5, 0.0))
        self.assertTrue(dense.adequacy(0.001, 0.04)["ok"])


class TestCorrection(unittest.TestCase):
    def test_roundtrip_grid(self):
        res = run({"scene": "grid"})
        s = res["stats"]
        self.assertTrue(s["pose_report"]["ok"])
        self.assertTrue(s["converged"])
        self.assertGreater(s["valid_fraction"], 0.9)
        self.assertLess(s["control_points"]["mean_px"], 0.2)
        self.assertLess(s["control_points"]["max_px"], 0.6)
        self.assertLess(s["mae_flat"], 0.02)
        self.assertLess(s["mae_valid"], 0.6 * s["mae_uncorrected"])
        self.assertLess(s["mae_valid"], s["mae_global_homography"])

    def test_all_directions(self):
        for d in ("top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left"):
            res = run({"scene": "grid", "direction": d})
            s = res["stats"]
            self.assertLess(s["control_points"]["mean_px"], 0.3, d)
            self.assertLess(s["mae_flat"], 0.025, d)

    def test_lines_straightened(self):
        res = run({"scene": "lines", "motion": "yaw"})
        w = res["stats"]["line_wobble"]
        self.assertIsNotNone(w)
        self.assertGreater(w["input_px"], 1.0)
        self.assertLess(w["corrected_px"], 0.35)
        self.assertLess(w["corrected_px"], 0.3 * w["input_px"])

    def test_offset_sensitivity(self):
        good = run({"scene": "grid", "t_offset_ms": pipeline.TRUE_OFFSET_S * 1000})
        bad = run({"scene": "grid", "t_offset_ms": pipeline.TRUE_OFFSET_S * 1000 + 4.0})
        self.assertLess(good["stats"]["mae_flat"], 0.02)
        self.assertGreater(bad["stats"]["mae_flat"], 3.0 * good["stats"]["mae_flat"])

    def test_folding_and_bounds_flagged(self):
        normal = run({"scene": "grid", "motion": "yaw"})
        self.assertEqual(normal["stats"]["reason_counts"]["folding"], 0)
        wild = run({"scene": "grid", "motion": "extreme", "readout_ms": 60.0,
                    "gyro_hz": 1000.0})
        self.assertGreater(wild["stats"]["reason_counts"]["folding"], 0)
        self.assertGreater(wild["stats"]["reason_counts"]["out_of_bounds"], 0)

    def test_detect_folding_synthetic(self):
        uu, vv = np.meshgrid(np.arange(50.0), np.arange(40.0))
        m = np.stack([uu, vv], axis=-1)
        self.assertFalse(detect_folding(m).any())
        m2 = m.copy()
        m2[..., 0] = uu[:, ::-1]
        self.assertTrue(detect_folding(m2).any())

    def test_pose_insufficiency_flagged_in_map(self):
        res = pipeline.run_synthetic({**FAST, "scene": "grid", "gyro_hz": 8.0})
        self.assertFalse(res["stats"]["pose_report"]["ok"])
        self.assertIn("sparse", {p["type"] for p in res["stats"]["pose_report"]["problems"]})

    def test_t_ref_outside_coverage(self):
        res = run({"scene": "grid", "t_ref_ms": 200.0})
        self.assertFalse(res["stats"]["pose_report"]["ok"])
        self.assertIn("t_ref_outside",
                      {p["type"] for p in res["stats"]["pose_report"]["problems"]})

    def test_plane_model(self):
        good = run({"scene": "plane", "model": "plane"})
        self.assertLess(good["stats"]["control_points"]["mean_px"], 0.3)
        self.assertLess(good["stats"]["mae_flat"], 0.02)
        bad = run({"scene": "plane", "model": "rotation"})
        self.assertGreater(bad["stats"]["mae_flat"], 1.5 * good["stats"]["mae_flat"])

    def test_moving_object_marked(self):
        res = run({"scene": "grid", "moving": True})
        s = res["stats"]
        self.assertGreater(s["reason_counts"]["independent_motion"], 1000)
        self.assertIsNotNone(s["moving_note"])
        mask = res["moving_mask"]
        diff = np.abs(res["images"]["corrected"] - res["images"]["reference"]).sum(-1)
        inside = pipeline._erode(mask, 3)
        self.assertGreater(diff[inside].mean(), diff[res["inv"].valid & ~mask].mean())


class TestExport(unittest.TestCase):
    def test_versioned_export_and_reconstruction(self):
        res = run({"scene": "grid"})
        tmp = tempfile.mkdtemp()
        try:
            exp = Exporter(tmp)
            from rs import imaging
            out = exp.export("t", res["params"],
                             {"corrected": res["images"]["corrected"],
                              "validity": imaging.validity_image(res["inv"].reasons).astype(float) / 255},
                             res["inv"], res["track_corr"], res["K"])
            ver = out["version"]
            self.assertEqual(ver, res["params"].version())
            for f in out["files"]:
                self.assertTrue(os.path.exists(f), f)
            self.assertTrue(any(f"t_corrected_{ver}.png" in f for f in out["files"]))
            self.assertTrue(any(f"t_map_{ver}.npz" in f for f in out["files"]))

            z = np.load(os.path.join(tmp, f"t_map_{ver}.npz"), allow_pickle=False)
            params = Params.from_dict(json.loads(str(z["params_json"])))
            self.assertEqual(params.version(), ver)
            track = PoseTrack(z["pose_times_s"], z["pose_quats_xyzw"])
            K = Intrinsics(*z["K"][:4])
            rd = params.readout
            readout = ReadoutSpec(params.camera["width_px"], params.camera["height_px"],
                                  rd["readout_time_s"], rd["direction"],
                                  rd["t0_s"], rd["t_offset_s"])
            scene = RotationScene(track, track.interp(np.array(z["t_ref_s"]))[0], K)
            inv2 = build_inverse_map(K, readout, scene, float(z["t_ref_s"]), track)
            self.assertTrue(np.allclose(inv2.uv, z["map_uv"], atol=1e-6))
            self.assertTrue(np.array_equal(inv2.reasons, z["reasons"]))

            with open(os.path.join(tmp, "manifest.json")) as f:
                man = json.load(f)
            self.assertEqual(man[0]["version"], ver)
        finally:
            shutil.rmtree(tmp)

    def test_version_changes_with_params(self):
        a = run({"scene": "grid", "t_offset_ms": 2.0})
        b = run({"scene": "grid", "t_offset_ms": 2.5})
        self.assertNotEqual(a["params"].version(), b["params"].version())


if __name__ == "__main__":
    unittest.main(verbosity=2)
