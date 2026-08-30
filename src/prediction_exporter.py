import json
import math
import os
from typing import Any

import numpy as np


COCO17_NUM_KEYPOINTS = 17


def _finite_float_or_none(value: Any):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def skeleton_dict_to_coco17(skeleton_3d: dict, one_based_ids: bool = True):
    """Converte {kp_id: [X,Y,Z]} para lista fixa COCO17.

    Joints ausentes viram ``None`` para não confundir ausência com [0, 0, 0].
    """
    keypoints = [None] * COCO17_NUM_KEYPOINTS

    if not isinstance(skeleton_3d, dict):
        return keypoints

    for raw_kp_id, xyz in skeleton_3d.items():
        try:
            kp_id = int(raw_kp_id)
        except (TypeError, ValueError):
            continue

        idx = kp_id - 1 if one_based_ids else kp_id
        if not (0 <= idx < COCO17_NUM_KEYPOINTS):
            continue

        arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            continue

        keypoints[idx] = [float(arr[0]), float(arr[1]), float(arr[2])]

    return keypoints


def sanitize_reprojection(reprojection: dict | None):
    if not isinstance(reprojection, dict):
        return None

    return {
        "mean_px": _finite_float_or_none(reprojection.get("mean_px")),
        "median_px": _finite_float_or_none(reprojection.get("median_px")),
        "rmse_px": _finite_float_or_none(reprojection.get("rmse_px")),
        "p95_px": _finite_float_or_none(reprojection.get("p95_px")),
        "num_points": int(reprojection.get("num_points", 0) or 0),
    }


def serialize_person(person: dict, one_based_ids: bool = True):
    matched_2d = person.get("matche_2d", {})

    return {
        "id": int(person.get("id", -1)),
        "keypoints_3d": skeleton_dict_to_coco17(
            person.get("skeleton_3d", {}),
            one_based_ids=one_based_ids,
        ),
        "matched_2d": {
            str(int(cam_idx)): int(skeleton_id)
            for cam_idx, skeleton_id in matched_2d.items()
        },
        "reprojection": sanitize_reprojection(person.get("reprojection")),
    }


class PredictionJSONWriter:
    """Acumula e salva as reconstruções 3D de forma atômica."""

    def __init__(
        self,
        output_path: str,
        dataset_name: str,
        camera_ids: list[int],
        use_cycle_consistency: bool,
        coordinate_unit: str = "calibration_native",
        save_every: int = 100,
        one_based_keypoint_ids: bool = True,
    ):
        self.output_path = output_path
        self.save_every = max(0, int(save_every))
        self.one_based_keypoint_ids = one_based_keypoint_ids
        self._frames_since_save = 0

        self.data = {
            "format_version": 1,
            "dataset": dataset_name,
            "keypoint_format": "COCO17",
            "num_keypoints": COCO17_NUM_KEYPOINTS,
            "coordinate_system": "world",
            "coordinate_unit": coordinate_unit,
            "camera_ids": [int(v) for v in camera_ids],
            "use_cycle_consistency": bool(use_cycle_consistency),
            "frames": {},
        }

    def add_frame(self, frame_idx: int, persons: list[dict], frame_paths=None):
        source_images = []
        if frame_paths is not None:
            source_images = [
                os.path.basename(path) if path else None
                for path in frame_paths
            ]

        self.data["frames"][str(int(frame_idx))] = {
            "source_images": source_images,
            "persons": [
                serialize_person(
                    person,
                    one_based_ids=self.one_based_keypoint_ids,
                )
                for person in persons
            ],
        }

        self._frames_since_save += 1
        if self.save_every and self._frames_since_save >= self.save_every:
            self.save()

    def save(self):
        parent = os.path.dirname(os.path.abspath(self.output_path))
        os.makedirs(parent, exist_ok=True)

        tmp_path = self.output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                self.data,
                f,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

        os.replace(tmp_path, self.output_path)
        self._frames_since_save = 0
