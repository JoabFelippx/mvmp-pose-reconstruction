import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skeleton_matcher import SkeletonMatcher
from skeleton_tracker_main import _build_matcher_params


class DatasetPipelineTests(unittest.TestCase):
    def test_distance_affinity_uses_dataset_d0(self):
        config = json.loads((ROOT / "etc/config.json").read_text())
        for dataset, expected in (("campus", 1.25), ("shelf", 0.25)):
            with self.subTest(dataset=dataset):
                params = _build_matcher_params(config["datasets"][dataset]["distance_d0"])
                matcher = SkeletonMatcher({}, params, 2, 18)
                self.assertEqual(matcher.distance_d0, expected)
                skeletons = [[np.ones((18, 2))], [np.ones((18, 2))]]
                with patch.object(matcher, "_ground_center", side_effect=[
                    np.array([0.0, 0.0]), np.array([expected, 0.0])
                ]):
                    affinity = matcher.distance_affinity(skeletons, {0: 0, 1: 1})
                self.assertAlmostEqual(affinity[0, 1], 0.5)
                self.assertAlmostEqual(affinity[1, 0], 0.5)

    def test_missing_ground_projection_preserves_matching_and_cycle_option(self):
        matcher = SkeletonMatcher({}, _build_matcher_params(), 2, 18)
        skeletons = [[np.ones((18, 2))], [np.ones((18, 2))]]
        for cycle in (False, True):
            with self.subTest(cycle=cycle), patch.object(
                matcher, "simple_match", return_value=np.array([[0.0, 1.0], [1.0, 0.0]])
            ), patch.object(
                matcher, "_validate_with_cycle_consistency",
                wraps=matcher._validate_with_cycle_consistency,
            ) as refinement:
                groups = matcher.match_skeletons(skeletons, [[5], [9]], cycle)
                self.assertEqual(groups, [{"ids": {0: 5, 1: 9}}])
                self.assertEqual(refinement.call_count, int(cycle))

    def test_precomputed_reconstruction_without_yolo_dependencies(self):
        config = json.loads((ROOT / "etc/config.json").read_text())
        config.pop("yolo_model")  # O modo pré-computado também dispensa essa seção.
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "etc").mkdir()
            calibs = work / "calibs"
            calibs.mkdir()
            detections = work / "detections"
            detections.mkdir()
            frames = work / "frames"
            intrinsic = np.array([[50.0, 0.0, 64.0], [0.0, 50.0, 64.0], [0.0, 0.0, 1.0]])
            rotation = np.diag([1.0, -1.0, -1.0])
            for camera in range(2):
                camera_frames = frames / f"Camera{camera}"
                camera_frames.mkdir(parents=True)
                center = np.array([(-0.1 if camera == 0 else 0.1), 0.0, 3.0])
                rt = np.column_stack((rotation, -rotation @ center))
                np.savez(calibs / f"calib_rt{camera}.npz", K=intrinsic, nK=intrinsic,
                         rt=rt, dist=np.zeros(5))
                annotations = {}
                for frame in range(2):
                    cv2.imwrite(str(camera_frames / f"{frame:04d}.jpg"),
                                np.zeros((128, 128, 3), dtype=np.uint8))
                    points = np.column_stack((np.linspace(-0.15, 0.15, 17),
                                              np.full(17, 0.2), np.ones(17)))
                    points[15:, 2] = 0.0
                    points[:, 0] += frame * 0.01
                    projected = (intrinsic @ rt @ np.column_stack((points, np.ones(17))).T).T
                    pixels = projected[:, :2] / projected[:, 2:]
                    annotations[str(frame)] = [{
                        "id": 0, "score": 1.0, "keypoints": [
                            {"id": i + 1, "x": float(x), "y": float(y), "score": 1.0}
                            for i, (x, y) in enumerate(pixels)
                        ],
                    }]
                (detections / f"camera_{camera}.json").write_text(json.dumps({"frames": annotations}))

            for dataset in ("campus", "shelf"):
                config["datasets"][dataset].update(
                    calib_path=str(calibs), data_path=str(frames), num_cameras=2,
                    apply_undistort=False,
                )
            (work / "etc/config.json").write_text(json.dumps(config))
            script = '''
import importlib.abc
import sys

class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "ultralytics", "skeletons", "visualizer", "vispy", "PyQt6"}:
            raise AssertionError("Import opcional indevido: " + fullname)

sys.meta_path.insert(0, BlockOptionalImports())
sys.path.insert(0, sys.argv.pop(1))
from skeleton_tracker_main import main
main()
'''
            for dataset, d0 in (("campus", 1.25), ("shelf", 0.25)):
                with self.subTest(dataset=dataset):
                    output = work / f"{dataset}.json"
                    result = subprocess.run([
                        sys.executable, "-c", script, str(ROOT / "src"),
                        "--source", "dataset", "--dataset_name", dataset,
                        "--input_2d", "precomputed", "--detections_2d", str(detections),
                        "--no_visualization", "--save_3d_json", str(output),
                    ], cwd=work, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f"Distance d0         : {d0}", result.stdout)
                    predictions = json.loads(output.read_text())
                    persons = [predictions["frames"][str(frame)]["persons"] for frame in range(2)]
                    self.assertTrue(all(len(frame) == 1 for frame in persons))
                    self.assertEqual(persons[0][0]["id"], persons[1][0]["id"])


if __name__ == "__main__":
    unittest.main()
