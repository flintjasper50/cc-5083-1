import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import build_examples
import check_params
import make_calibration
from rs import Params, PoseTrack, imaging, pipeline


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)


class TestMakeCalibration(TempDirCase):
    def test_samples_load_and_correct(self):
        written = make_calibration.build_sample(self.tmp)
        self.assertEqual(len(written), 6)
        params = Params.load(os.path.join(self.tmp, "sample_params.json"))
        with open(os.path.join(self.tmp, "sample_poses.json")) as f:
            track = PoseTrack.from_dict(json.load(f))
        img = imaging.load_image(os.path.join(self.tmp, "sample_capture.png"))
        corrected, inv, stats = pipeline.run_photo(img, params, track)
        self.assertTrue(stats["pose_report"]["ok"])
        self.assertGreater(stats["valid_fraction"], 0.9)
        # 样例参数与拍摄设定一致：校正后平坦区应接近参考图
        ref = imaging.load_image(os.path.join(self.tmp, "sample_reference.png"))
        diff = np.abs(corrected - ref).sum(-1)
        gray = ref.mean(axis=-1)
        gx, gy = np.gradient(gray)
        flat = (np.hypot(gx, gy) < 0.02) & inv.valid
        self.assertLess(float(diff[flat].mean()), 0.02)

    def test_samples_pass_check_params(self):
        make_calibration.build_sample(self.tmp)
        problems = []
        check_params.check_params_dict(
            json.load(open(os.path.join(self.tmp, "sample_params.json"))),
            "sample_params.json", problems)
        self.assertEqual(problems, [])


class TestCheckParams(TempDirCase):
    def _base(self):
        make_calibration.build_sample(self.tmp)
        with open(os.path.join(self.tmp, "sample_params.json")) as f:
            return json.load(f)

    def problems_of(self, d):
        problems = []
        check_params.check_params_dict(d, "test.json", problems)
        return [m for _, m in problems]

    def test_missing_t_ref_is_an_error(self):
        d = self._base()
        del d["t_ref_s"]
        msgs = self.problems_of(d)
        self.assertTrue(any("t_ref_s" in m for m in msgs), msgs)

    def test_bad_direction_and_units(self):
        d = self._base()
        d["readout"]["direction"] = "up_down"
        self.assertTrue(any("direction" in m for m in self.problems_of(d)))
        d = self._base()
        d["motion"]["omega_rad_s"] = [0.0, 90.0, 0.0]  # 疑似度/秒
        self.assertTrue(any("弧度" in m for m in self.problems_of(d)))
        d = self._base()
        d["units"]["angle"] = "deg"
        self.assertTrue(any("弧度" in m for m in self.problems_of(d)))

    def test_stale_plane_keys_flagged(self):
        d = self._base()
        d["scene"] = {"model": "plane", "normal": [0, 0, 1], "dist": 2.0,
                      "center0": [0, 0, 0], "velocity": [0, 0, 0]}
        self.assertTrue(any("dist_m" in m for m in self.problems_of(d)))

    def test_manifest_detects_stale_and_missing(self):
        make_calibration.build_sample(self.tmp)
        out_dir = os.path.join(self.tmp, "out")
        img = imaging.load_image(os.path.join(self.tmp, "sample_capture.png"))
        params = Params.load(os.path.join(self.tmp, "sample_params.json"))
        with open(os.path.join(self.tmp, "sample_poses.json")) as f:
            track = PoseTrack.from_dict(json.load(f))
        corrected, inv, stats = pipeline.run_photo(img, params, track)
        from rs import Exporter, Intrinsics
        K = Intrinsics(params.camera["fx_px"], params.camera["fy_px"],
                       params.camera["cx_px"], params.camera["cy_px"])
        Exporter(out_dir).export("t", params, {"corrected": corrected}, inv, track, K)

        problems = []
        check_params.check_manifest(os.path.join(out_dir, "manifest.json"), problems)
        self.assertEqual(problems, [])

        ver = params.version()
        os.remove(os.path.join(out_dir, f"t_corrected_{ver}.png"))
        with open(os.path.join(out_dir, f"t_params_{ver}.json")) as f:
            d = json.load(f)
        d["readout"]["readout_time_s"] = 0.05  # 篡改后版本标签即失效
        with open(os.path.join(out_dir, f"t_params_{ver}.json"), "w") as f:
            json.dump(d, f)
        problems = []
        check_params.check_manifest(os.path.join(out_dir, "manifest.json"), problems)
        msgs = [m for _, m in problems]
        self.assertTrue(any("不存在" in m for m in msgs), msgs)
        self.assertTrue(any("已失效" in m for m in msgs), msgs)


class TestBuildExamples(TempDirCase):
    def test_single_example_fresh_dir(self):
        out_dir = os.path.join(self.tmp, "out")
        docs = os.path.join(self.tmp, "docs")
        readme = os.path.join(self.tmp, "README.md")
        with open(readme, "w") as f:
            f.write("前\n%s\n旧内容\n%s\n后\n" % (build_examples.METRICS_BEGIN,
                                                build_examples.METRICS_END))
        rows, failures = build_examples.build(
            out_dir, docs, readme, only={"grid_static"}, size=(160, 120))
        self.assertEqual(failures, [])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertLess(row["cp_mean_px"], 0.05)
        self.assertTrue(os.path.isfile(os.path.join(
            docs, "figures", "grid_static_corrected.png")))
        self.assertTrue(os.path.isfile(os.path.join(
            docs, "figures", "grid_static_row_timing.png")))
        with open(readme) as f:
            text = f.read()
        self.assertIn("grid_static", text)
        self.assertNotIn("旧内容", text)
        with open(os.path.join(docs, "metrics.json")) as f:
            self.assertEqual(json.load(f)[0]["name"], "grid_static")

    def test_control_point_errors_read_exported_map(self):
        out_dir = os.path.join(self.tmp, "out")
        docs = os.path.join(self.tmp, "docs")
        rows, failures = build_examples.build(
            out_dir, docs, os.path.join(self.tmp, "no_readme.md"),
            only={"grid_mixed"}, size=(160, 120))
        self.assertEqual(failures, [])
        ver = rows[0]["version"]
        errs = build_examples.control_point_errors_from_export(
            os.path.join(out_dir, f"grid_mixed_map_{ver}.npz"),
            build_examples.scene_control_points({"scene": "grid"}, 160, 120))
        self.assertTrue(errs)
        self.assertLess(float(np.mean(errs)), 0.3)


class TestCliAndUi(TempDirCase):
    def test_demo_photo_cli(self):
        make_calibration.build_sample(self.tmp)
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "demo.py"), "photo",
             "--image", os.path.join(self.tmp, "sample_capture.png"),
             "--params", os.path.join(self.tmp, "sample_params.json"),
             "--poses", os.path.join(self.tmp, "sample_poses.json"),
             "--out", os.path.join(self.tmp, "out")],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        stats = json.loads(r.stdout.split("\n导出版本")[0])
        self.assertGreater(stats["valid_fraction"], 0.9)

    def test_demo_photo_cli_missing_t_ref(self):
        make_calibration.build_sample(self.tmp)
        with open(os.path.join(self.tmp, "sample_params.json")) as f:
            d = json.load(f)
        del d["t_ref_s"]
        bad = os.path.join(self.tmp, "bad_params.json")
        with open(bad, "w") as f:
            json.dump(d, f)
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "demo.py"), "photo",
             "--image", os.path.join(self.tmp, "sample_capture.png"),
             "--params", bad,
             "--poses", os.path.join(self.tmp, "sample_poses.json"),
             "--out", os.path.join(self.tmp, "out")],
            capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("t_ref_s", r.stderr + r.stdout)

    def test_ui_photo_endpoint(self):
        import base64
        sys.path.insert(0, os.path.join(ROOT, "ui"))
        import server
        make_calibration.build_sample(self.tmp)
        with open(os.path.join(self.tmp, "sample_capture.png"), "rb") as f:
            data_url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        with open(os.path.join(self.tmp, "sample_params.json")) as f:
            params = json.load(f)
        with open(os.path.join(self.tmp, "sample_poses.json")) as f:
            poses = json.load(f)
        body = {"image": data_url, "params": params, "poses": poses,
                "overrides": {"readout_ms": 32.0, "t_offset_ms": 2.0,
                              "t_ref_ms": 16.0, "direction": "top_to_bottom"}}
        resp = server._photo_response(body)
        self.assertGreater(resp["stats"]["valid_fraction"], 0.9)
        self.assertTrue(resp["images"]["corrected"].startswith("data:image/png"))

    def test_ui_photo_missing_t_ref_rejected(self):
        sys.path.insert(0, os.path.join(ROOT, "ui"))
        import server
        make_calibration.build_sample(self.tmp)
        with open(os.path.join(self.tmp, "sample_params.json")) as f:
            params = json.load(f)
        del params["t_ref_s"]
        with self.assertRaises(ValueError) as ctx:
            server._photo_response({"image": "", "params": params, "poses": {}})
        self.assertIn("t_ref_s", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
