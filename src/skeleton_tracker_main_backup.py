import numpy as np
import os
import json
import cv2

from config import cfg
from is_utils.stream_handler import StreamHandler
from fundamental_matrices import FundamentalMatrices
from skeleton_matcher import SkeletonMatcher
from reconstructor_3d import Reconstructor3D
from visualizer import Visualizer
from utils import get_skeleton_center
from is_utils.streamChannel import StreamChannel
from is_wire.core import Message
from is_msgs.image_pb2 import Image


def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def calibration_files_path(calib_path: str, num_cameras: int):
    return [f"{calib_path}{i}.npz" for i in range(1, num_cameras + 1)]


def _to_image(image, encode_format: str = ".jpeg", compression_level: float = 0.8) -> Image:
    if encode_format == ".jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * 100)]
    elif encode_format == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * 9)]
    else:
        return Image()
    cimage = cv2.imencode(ext=encode_format, img=image, params=params)
    return Image(data=cimage[1].tobytes())


def main():
    print("=== Skeleton Tracker 3D ===")
    print(f"  broker_uri           : {cfg.broker_uri}")
    print(f"  num_cameras          : {cfg.num_cameras}")
    print(f"  calib_path           : {cfg.calib_path}")
    print(f"  use_undistorted      : {cfg.use_undistorted}")
    print(f"  num_keypoints        : {cfg.num_keypoints}")
    print(f"  use_cycle_consistency: {cfg.use_cycle_consistency}")

    camera_files_path = calibration_files_path(cfg.calib_path, cfg.num_cameras)
    calib_files_data  = [np.load(f) for f in camera_files_path]

    matcher_params = {
        "use_cycle_consistency":    cfg.use_cycle_consistency,
        "sigma_tolerance":          cfg.sigma_tolerance,
        "max_error_per_joint":      cfg.max_error_per_joint,
        "weight_quality":           cfg.weight_quality,
        "weight_quantity":          cfg.weight_quantity,
        "min_compatibility_score":  cfg.min_compatibility_score,
        "top_k_cycle_candidates":   cfg.top_k_cycle_candidates,
        "early_stop_cycle_score":   cfg.early_stop_cycle_score,
        "weight_cycle":             cfg.weight_cycle,
        "min_cycle_score":          cfg.min_cycle_score,
        "max_intersection_dist":    cfg.max_intersection_dist,
        "weight_distance":          cfg.weight_distance,
        "weight_score":             cfg.weight_score,
        "min_keypoints_for_grouping": cfg.min_keypoints_for_grouping,
        "min_kp_ratio":             cfg.min_kp_ratio,
        "min_fallback_score":       cfg.min_fallback_score,
        "kp_weights":               cfg.kp_weights,
    }

    print("Inicializando módulos...")
    geometry              = FundamentalMatrices(camera_files_path, use_undistorted=cfg.use_undistorted)
    projection_matrices   = geometry.projection_matrices_all()
    fundamentals          = geometry.fundamental_matrices_all()
    all_calibs_parameters = geometry.get_all_calibs_parameters()

    stream       = StreamHandler(cfg.broker_uri, cfg.num_cameras, cfg.num_keypoints, all_calibs_parameters)
    channel      = StreamChannel(cfg.broker_uri)
    matcher      = SkeletonMatcher(fundamentals, matcher_params, cfg.num_cameras, cfg.num_keypoints)
    reconstructor = Reconstructor3D(projection_matrices, cfg.num_cameras, cfg.num_keypoints)
    visualizer   = Visualizer(all_calibs_parameters)

    print("Loop principal iniciado.")
    while True:
        raw_messages = stream.get_latest_messages()
        if raw_messages is None:
            continue

        annotations = stream.prepare_input_data(raw_messages, calib_files_data)

        current_detections_3d = []
        skeletons_by_detection = []
        matched_2d_persons = []
        skeletons_to_visualize = []

        skeletons_2d, ids_2d = matcher.extract_skeletons_from_annotations(annotations)
        matched_persons       = matcher.match_skeletons(skeletons_2d, ids_2d, [1])
        reconstructed_skeletons = reconstructor.reconstruct_all(matched_persons, annotations)

        for idx, skeleton_data in enumerate(reconstructed_skeletons):
            if not skeleton_data:
                continue

            hip_center = get_skeleton_center(skeleton_data)
            if hip_center is not None:
                center_point = hip_center
            else:
                points_3d = np.array(list(skeleton_data.values()))
                if len(points_3d) == 0:
                    continue
                center_point = np.mean(points_3d, axis=0)

            current_detections_3d.append(center_point)
            skeletons_by_detection.append(skeleton_data)
            matched_2d_persons.append(matched_persons[idx]['ids'])
            skeletons_to_visualize.append({
                "id": idx + 1,
                "skeleton_3d": skeleton_data,
                "average_point": 0,
                "matche_2d": 0,
            })

        plot_img_bgr = visualizer.update(skeletons_to_visualize)

        rendered_msg = Message()
        rendered_msg.topic = "SkeletonDetector.3D"
        rendered_msg.pack(_to_image(plot_img_bgr))
        channel.publish(rendered_msg)

        skt_msg = Message()
        skt_msg.topic = "SkeletonDetector.3D.Annotations"
        skt_msg.body  = json.dumps(skeletons_to_visualize, default=numpy_to_list).encode("utf-8")
        channel.publish(skt_msg)


if __name__ == "__main__":
    main()