#!/usr/bin/env python3
"""Avaliação PCP3D para Shelf/Campus usando annotation_3d.json como GT.

Entradas
--------
1) JSON de predições gerado pelo pipeline da IC:
   {
       "keypoint_format": "COCO17",
       "frames": {
           "350": {
               "persons": [
                   {"keypoints_3d": [[x,y,z], ...]}
               ]
           }
       }
   }

2) GT annotation_3d.json:
   [
       {
           "timestamp": 0.0,
           "poses": [
               {
                   "id": 0,
                   "points_3d": [[x,y,z], ... 14 joints ...],
                   "scores": [...]
               }
           ]
       },
       ...
   ]

Métrica principal
-----------------
PCP3D com alpha=0.5 no protocolo Shelf/Campus.

Diagnósticos adicionais
-----------------------
- Recall@500mm
- MPJPE da predição mais próxima de cada GT

Observação
----------
O annotation_3d.json já está no esqueleto de 14 joints de Shelf/Campus.
As predições da IC estão em COCO17 e são convertidas para esses 14 joints.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict

import numpy as np


COCO_TO_DATASET_12 = np.array(
    [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9],
    dtype=np.int64,
)

# 9 membros explícitos. O torso é calculado separadamente como a 10ª parte.
PCP_LIMBS = [
    [0, 1],    # lower right leg
    [1, 2],    # upper right leg
    [3, 4],    # upper left leg
    [4, 5],    # lower left leg
    [6, 7],    # lower right arm
    [7, 8],    # upper right arm
    [9, 10],   # upper left arm
    [10, 11],  # lower left arm
    [12, 13],  # head
]

BONE_GROUPS = OrderedDict([
    ("Head", [8]),
    ("Torso", [9]),
    ("Upper arms", [5, 6]),
    ("Lower arms", [4, 7]),
    ("Upper legs", [1, 2]),
    ("Lower legs", [0, 3]),
])

OFFICIAL_FRAME_RANGES = {
    "shelf": list(range(300, 601)),
    "campus": list(range(350, 471)) + list(range(650, 751)),
}


def load_predictions(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("keypoint_format") != "COCO17":
        raise ValueError(
            "O JSON de predição precisa ter keypoint_format='COCO17'. "
            f"Recebido: {data.get('keypoint_format')!r}"
        )

    return data


def load_gt_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "annotation_3d.json deve ser uma lista de frames."
        )

    return data


def unit_scale_to_mm(unit: str) -> float:
    scales = {
        "m": 1000.0,
        "cm": 10.0,
        "mm": 1.0,
    }
    return scales[unit]


def resolve_prediction_scale_to_mm(
    pred_unit: str,
    explicit_scale: float | None,
) -> float:
    if explicit_scale is not None:
        if explicit_scale <= 0:
            raise ValueError("--pred-scale-to-mm deve ser > 0")
        return float(explicit_scale)

    if pred_unit == "native":
        raise ValueError(
            "--pred-unit native exige --pred-scale-to-mm. "
            "Use, por exemplo, --pred-unit cm se a reconstrução estiver em centímetros."
        )

    return unit_scale_to_mm(pred_unit)


def get_num_gt_actors(gt_frames: list) -> int:
    max_id = -1

    for frame in gt_frames:
        for pose in frame.get("poses", []):
            try:
                actor_id = int(pose["id"])
            except (KeyError, TypeError, ValueError):
                continue
            max_id = max(max_id, actor_id)

    return max_id + 1


def gt_pose_at(
    gt_frames: list,
    actor_id: int,
    frame_idx: int,
    gt_scale_to_mm: float,
):
    """Retorna GT [14,3] em mm para actor_id/frame_idx."""

    if frame_idx < 0 or frame_idx >= len(gt_frames):
        return None

    frame = gt_frames[frame_idx]

    for pose in frame.get("poses", []):
        try:
            pose_id = int(pose["id"])
        except (KeyError, TypeError, ValueError):
            continue

        if pose_id != actor_id:
            continue

        points = pose.get("points_3d")
        if points is None:
            return None

        arr = np.asarray(points, dtype=np.float64)

        if arr.shape != (14, 3):
            return None

        if not np.all(np.isfinite(arr)):
            return None

        return arr * gt_scale_to_mm

    return None


def json_person_to_coco17(
    person: dict,
    scale_to_mm: float,
):
    raw = person.get("keypoints_3d", [])
    coco = np.full((17, 3), np.nan, dtype=np.float64)

    for idx in range(min(17, len(raw))):
        xyz = raw[idx]

        if xyz is None:
            continue

        arr = np.asarray(xyz, dtype=np.float64).reshape(-1)

        if arr.size < 3:
            continue

        if not np.all(np.isfinite(arr[:3])):
            continue

        coco[idx] = arr[:3] * scale_to_mm

    return coco


def coco_to_campus3d(coco_pose: np.ndarray):
    """COCO17 -> Campus14 seguindo a conversão usada na literatura."""

    campus_pose = np.full((14, 3), np.nan, dtype=np.float64)
    campus_pose[0:12] = coco_pose[COCO_TO_DATASET_12]

    # COCO:
    # 3 = left ear
    # 4 = right ear
    # 5 = left shoulder
    # 6 = right shoulder
    mid_shoulder = (
        coco_pose[5] + coco_pose[6]
    ) / 2.0

    head_center = (
        coco_pose[3] + coco_pose[4]
    ) / 2.0

    head_bottom = (
        mid_shoulder + head_center
    ) / 2.0

    head_top = (
        head_bottom
        + (head_center - head_bottom) * 2.0
    )

    campus_pose[12] = head_bottom
    campus_pose[13] = head_top

    return campus_pose


def coco_to_shelf3d(coco_pose: np.ndarray):
    """COCO17 -> Shelf14 seguindo a conversão usada no protocolo VoxelPose."""

    shelf_pose = np.full((14, 3), np.nan, dtype=np.float64)
    shelf_pose[0:12] = coco_pose[COCO_TO_DATASET_12]

    mid_shoulder = (
        coco_pose[5] + coco_pose[6]
    ) / 2.0

    head_center = (
        coco_pose[3] + coco_pose[4]
    ) / 2.0

    head_bottom_aux = (
        mid_shoulder + head_center
    ) / 2.0

    head_top_aux = (
        head_bottom_aux
        + (head_center - head_bottom_aux) * 2.0
    )

    # Conversão específica do Shelf usada no VoxelPose.
    shelf_pose[12] = (
        shelf_pose[8] + shelf_pose[9]
    ) / 2.0

    shelf_pose[13] = coco_pose[0]

    shelf_pose[13] = (
        shelf_pose[12]
        + (shelf_pose[13] - shelf_pose[12])
        * np.array([0.75, 0.75, 1.5])
    )

    shelf_pose[12] = (
        shelf_pose[12]
        + (coco_pose[0] - shelf_pose[12])
        * np.array([0.5, 0.5, 0.5])
    )

    alpha = 0.75

    shelf_pose[13] = (
        shelf_pose[13] * alpha
        + head_top_aux * (1.0 - alpha)
    )

    shelf_pose[12] = (
        shelf_pose[12] * alpha
        + head_bottom_aux * (1.0 - alpha)
    )

    return shelf_pose


def convert_prediction(
    coco_pose: np.ndarray,
    dataset: str,
):
    if dataset == "campus":
        pose = coco_to_campus3d(coco_pose)

    elif dataset == "shelf":
        pose = coco_to_shelf3d(coco_pose)

    else:
        raise ValueError(dataset)

    if not np.all(np.isfinite(pose)):
        return None

    return pose


def frame_predictions(
    predictions_json: dict,
    frame_idx: int,
    dataset: str,
    scale_to_mm: float,
):
    frame = (
        predictions_json
        .get("frames", {})
        .get(str(frame_idx))
    )

    if not frame:
        return (
            np.empty((0, 14, 3), dtype=np.float64),
            0,
            0,
        )

    valid = []
    total = 0
    invalid = 0

    for person in frame.get("persons", []):
        total += 1

        coco = json_person_to_coco17(
            person,
            scale_to_mm,
        )

        converted = convert_prediction(
            coco,
            dataset,
        )

        if converted is None:
            invalid += 1
            continue

        valid.append(converted)

    if not valid:
        return (
            np.empty((0, 14, 3), dtype=np.float64),
            total,
            invalid,
        )

    return (
        np.stack(valid, axis=0),
        total,
        invalid,
    )


def pcp_part_is_correct(
    pred_s,
    pred_e,
    gt_s,
    gt_e,
    alpha=0.5,
):
    error_s = np.linalg.norm(
        pred_s - gt_s
    )

    error_e = np.linalg.norm(
        pred_e - gt_e
    )

    limb_length = np.linalg.norm(
        gt_s - gt_e
    )

    if limb_length <= 1e-12:
        return False

    return (
        (error_s + error_e) / 2.0
        <= alpha * limb_length
    )


def evaluate(
    dataset: str,
    predictions_json: dict,
    gt_frames: list,
    prediction_scale_to_mm: float,
    gt_scale_to_mm: float,
    recall_threshold: float = 500.0,
    alpha: float = 0.5,
    debug_frame: int | None = None,
):
    frame_range = OFFICIAL_FRAME_RANGES[dataset]
    num_person = get_num_gt_actors(gt_frames)

    if num_person == 0:
        raise RuntimeError(
            "Nenhum ator foi encontrado no annotation_3d.json."
        )

    correct_parts = np.zeros(
        num_person,
        dtype=np.float64,
    )

    total_parts = np.zeros(
        num_person,
        dtype=np.float64,
    )

    bone_correct_parts = np.zeros(
        (num_person, 10),
        dtype=np.float64,
    )

    total_gt = 0
    match_gt = 0

    nearest_mpjpes = []
    recalled_mpjpes = []

    total_predictions = 0
    invalid_predictions = 0
    missing_json_frames = 0

    debug_rows = []

    for frame_idx in frame_range:

        if (
            str(frame_idx)
            not in predictions_json.get("frames", {})
        ):
            missing_json_frames += 1

        pred, pred_total, pred_invalid = (
            frame_predictions(
                predictions_json,
                frame_idx,
                dataset,
                prediction_scale_to_mm,
            )
        )

        total_predictions += pred_total
        invalid_predictions += pred_invalid

        for actor_id in range(num_person):

            gt = gt_pose_at(
                gt_frames,
                actor_id,
                frame_idx,
                gt_scale_to_mm,
            )

            if gt is None:
                continue

            total_gt += 1

            if len(pred) == 0:
                total_parts[actor_id] += 10

                if debug_frame == frame_idx:
                    debug_rows.append(
                        (
                            actor_id,
                            None,
                            None,
                            "sem predicao valida",
                        )
                    )

                continue

            mpjpes = np.mean(
                np.linalg.norm(
                    gt[np.newaxis, :, :] - pred,
                    axis=-1,
                ),
                axis=-1,
            )

            min_n = int(
                np.argmin(mpjpes)
            )

            min_mpjpe = float(
                np.min(mpjpes)
            )

            nearest_mpjpes.append(
                min_mpjpe
            )

            if min_mpjpe < recall_threshold:
                match_gt += 1
                recalled_mpjpes.append(
                    min_mpjpe
                )

            if debug_frame == frame_idx:
                debug_rows.append(
                    (
                        actor_id,
                        min_n,
                        min_mpjpe,
                        "ok",
                    )
                )

            chosen = pred[min_n]

            for bone_idx, (start, end) in enumerate(
                PCP_LIMBS
            ):
                total_parts[actor_id] += 1

                if pcp_part_is_correct(
                    chosen[start],
                    chosen[end],
                    gt[start],
                    gt[end],
                    alpha=alpha,
                ):
                    correct_parts[actor_id] += 1

                    bone_correct_parts[
                        actor_id,
                        bone_idx,
                    ] += 1

            # 10ª parte: torso
            # centro dos quadris -> bottom head
            pred_hip = (
                chosen[2] + chosen[3]
            ) / 2.0

            gt_hip = (
                gt[2] + gt[3]
            ) / 2.0

            total_parts[actor_id] += 1

            if pcp_part_is_correct(
                pred_hip,
                chosen[12],
                gt_hip,
                gt[12],
                alpha=alpha,
            ):
                correct_parts[actor_id] += 1

                bone_correct_parts[
                    actor_id,
                    9,
                ] += 1

    actor_pcp = (
        correct_parts
        / (total_parts + 1e-8)
    )

    if len(actor_pcp) >= 3:
        avg_pcp = float(
            np.mean(actor_pcp[:3])
        )
    else:
        avg_pcp = float(
            np.mean(actor_pcp)
        )

    bone_person_pcp = OrderedDict()

    for group_name, indices in BONE_GROUPS.items():

        denom = (
            total_parts / 10.0
            * len(indices)
        )

        bone_person_pcp[group_name] = (
            np.sum(
                bone_correct_parts[:, indices],
                axis=-1,
            )
            / (denom + 1e-8)
        )

    recall = (
        match_gt
        / (total_gt + 1e-8)
    )

    results = {
        "dataset": dataset,
        "protocol": (
            "Shelf/Campus PCP3D "
            "with annotation_3d.json GT"
        ),
        "official_num_frames": len(frame_range),
        "frame_ranges": (
            [[300, 600]]
            if dataset == "shelf"
            else [[350, 470], [650, 750]]
        ),
        "alpha": float(alpha),
        "recall_threshold_mm": float(
            recall_threshold
        ),
        "prediction_scale_to_mm": float(
            prediction_scale_to_mm
        ),
        "gt_scale_to_mm": float(
            gt_scale_to_mm
        ),
        "num_gt_actors": int(
            num_person
        ),
        "actor_pcp": [
            float(v)
            for v in actor_pcp
        ],
        "avg_pcp_first_3_actors": (
            avg_pcp
        ),
        "bone_group_pcp": {
            name: [
                float(v)
                for v in values
            ]
            for name, values
            in bone_person_pcp.items()
        },
        "recall": float(
            recall
        ),
        "total_gt_poses": int(
            total_gt
        ),
        "matched_gt_under_threshold": int(
            match_gt
        ),
        "nearest_prediction_mpjpe_mm_mean": (
            float(np.mean(nearest_mpjpes))
            if nearest_mpjpes
            else None
        ),
        "nearest_prediction_mpjpe_mm_median": (
            float(np.median(nearest_mpjpes))
            if nearest_mpjpes
            else None
        ),
        "recalled_mpjpe_mm_mean": (
            float(np.mean(recalled_mpjpes))
            if recalled_mpjpes
            else None
        ),
        "total_json_predictions": int(
            total_predictions
        ),
        "discarded_incomplete_predictions": int(
            invalid_predictions
        ),
        "missing_official_frames_in_json": int(
            missing_json_frames
        ),
    }

    return results, debug_rows


def print_results(results: dict):
    print("\n=== Avaliacao Shelf/Campus ===")
    print(
        f"Dataset: {results['dataset']}"
    )
    print(
        f"Frames oficiais: "
        f"{results['official_num_frames']}"
    )
    print(
        f"Atores GT encontrados: "
        f"{results['num_gt_actors']}"
    )
    print(
        f"Escala pred -> mm: "
        f"{results['prediction_scale_to_mm']}"
    )
    print(
        f"Escala GT -> mm: "
        f"{results['gt_scale_to_mm']}"
    )
    print()

    for idx, pcp in enumerate(
        results["actor_pcp"]
    ):
        print(
            f"Actor {idx + 1}: "
            f"PCP3D = {pcp * 100:.2f}%"
        )

    print(
        "PCP3D medio (atores 1-3): "
        f"{results['avg_pcp_first_3_actors'] * 100:.2f}%"
    )

    print(
        f"Recall@"
        f"{results['recall_threshold_mm']:.0f}mm: "
        f"{results['recall'] * 100:.2f}%"
    )

    if (
        results[
            "nearest_prediction_mpjpe_mm_mean"
        ]
        is not None
    ):
        print(
            "MPJPE nearest (diagnostico): "
            f"{results['nearest_prediction_mpjpe_mm_mean']:.2f} mm"
        )

    print(
        "Predicoes incompletas descartadas: "
        f"{results['discarded_incomplete_predictions']}"
    )

    print(
        "Frames oficiais ausentes nas predicoes: "
        f"{results['missing_official_frames_in_json']}"
    )

    print("\nPCP3D por grupo e ator:")

    for group, values in (
        results["bone_group_pcp"].items()
    ):
        formatted = " | ".join(
            f"A{i + 1}: {v * 100:.2f}%"
            for i, v in enumerate(values)
        )

        print(
            f"  {group:<11} {formatted}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Avalia predicoes COCO17 "
            "contra annotation_3d.json "
            "de Shelf/Campus."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["shelf", "campus"],
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="JSON 3D gerado pelo pipeline da IC.",
    )

    parser.add_argument(
        "--gt",
        required=True,
        help=(
            "datasets\\...\\annotation_3d.json"
        ),
    )

    parser.add_argument(
        "--pred-unit",
        choices=["m", "cm", "mm", "native"],
        default="native",
        help=(
            "Unidade das coordenadas reconstruidas."
        ),
    )

    parser.add_argument(
        "--pred-scale-to-mm",
        type=float,
        default=None,
        help=(
            "Fator manual predicao -> mm. "
            "Sobrescreve --pred-unit."
        ),
    )

    parser.add_argument(
        "--gt-unit",
        choices=["m", "cm", "mm"],
        default="cm",
        help=(
            "Unidade do annotation_3d.json. "
            "Neste formato de Shelf/Campus, "
            "o padrao esperado e cm."
        ),
    )

    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=500.0,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Salva as metricas em JSON.",
    )

    parser.add_argument(
        "--debug-frame",
        type=int,
        default=None,
        help=(
            "Mostra GT->pred e MPJPE "
            "para um frame especifico."
        ),
    )

    args = parser.parse_args()

    prediction_scale_to_mm = (
        resolve_prediction_scale_to_mm(
            args.pred_unit,
            args.pred_scale_to_mm,
        )
    )

    gt_scale_to_mm = unit_scale_to_mm(
        args.gt_unit
    )

    predictions = load_predictions(
        args.predictions
    )

    gt_frames = load_gt_json(
        args.gt
    )

    results, debug_rows = evaluate(
        dataset=args.dataset,
        predictions_json=predictions,
        gt_frames=gt_frames,
        prediction_scale_to_mm=(
            prediction_scale_to_mm
        ),
        gt_scale_to_mm=(
            gt_scale_to_mm
        ),
        recall_threshold=(
            args.recall_threshold
        ),
        alpha=args.alpha,
        debug_frame=args.debug_frame,
    )

    print_results(results)

    if args.debug_frame is not None:
        print(
            f"\nDebug frame "
            f"{args.debug_frame}:"
        )

        if not debug_rows:
            print(
                "  Nenhum GT anotado "
                "nesse frame."
            )

        for (
            actor_id,
            pred_idx,
            mpjpe,
            status,
        ) in debug_rows:

            if mpjpe is None:
                print(
                    f"  GT actor "
                    f"{actor_id + 1}: "
                    f"{status}"
                )
            else:
                print(
                    f"  GT actor "
                    f"{actor_id + 1} "
                    f"-> pred {pred_idx}: "
                    f"MPJPE={mpjpe:.2f} mm"
                )

    if args.output:
        parent = os.path.dirname(
            os.path.abspath(args.output)
        )

        os.makedirs(
            parent,
            exist_ok=True,
        )

        with open(
            args.output,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

        print(
            f"\nMetricas salvas em: "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()
