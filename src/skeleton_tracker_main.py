import argparse
import json

import numpy as np
import cv2

from config import cfg
from fundamental_matrices import FundamentalMatrices
from skeleton_matcher import SkeletonMatcher
from reconstructor_3d import Reconstructor3D
from visualizer import Visualizer
from utils import create_adaptive_camera_grid


def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


# ──────────────────────────────────────────────────────────────────────────
# Caminhos de calibração — formato difere entre IS e dataset
# ──────────────────────────────────────────────────────────────────────────

def calibration_files_path_is(calib_path: str, num_cameras: int):
    """IS: prefixo direto + índice 1-based. Ex.: calib_rt1.npz"""
    return [f"{calib_path}{i}.npz" for i in range(1, num_cameras + 1)]


def calibration_files_path_dataset(calib_path: str, camera_ids: list[int]):
    """Dataset: pasta + calib_rt{cam_id}.npz, índice 0-based conforme camera_ids."""
    return [f"{calib_path}/calib_rt{cam_id}.npz" for cam_id in camera_ids]


def _parse_camera_ids(cameras_str: str, num_cameras: int) -> list[int]:
    ids: set[int] = set()
    for part in cameras_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            ids.update(range(int(start), int(end) + 1))
        elif part:
            ids.add(int(part))
    valid   = sorted(i for i in ids if 0 <= i < num_cameras)
    invalid = sorted(i for i in ids if not (0 <= i < num_cameras))
    if invalid:
        print(f"Aviso: IDs de câmera fora do intervalo ignorados: {invalid}")
    return valid


def _to_image(image, encode_format: str = ".jpeg", compression_level: float = 0.8):
    """Só usado no modo IS; import de is_msgs fica sob demanda para não exigir a lib no modo dataset."""
    from is_msgs.image_pb2 import Image
    if encode_format == ".jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * 100)]
    elif encode_format == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * 9)]
    else:
        return Image()
    cimage = cv2.imencode(ext=encode_format, img=image, params=params)
    return Image(data=cimage[1].tobytes())


def _build_matcher_params():
    return {
        "use_cycle_consistency":      cfg.use_cycle_consistency,
        "sigma_tolerance":            cfg.sigma_tolerance,
        "max_error_per_joint":        cfg.max_error_per_joint,
        "weight_quality":             cfg.weight_quality,
        "weight_quantity":            cfg.weight_quantity,
        "min_compatibility_score":    cfg.min_compatibility_score,
        "top_k_cycle_candidates":     cfg.top_k_cycle_candidates,
        "early_stop_cycle_score":     cfg.early_stop_cycle_score,
        "weight_cycle":               cfg.weight_cycle,
        "min_cycle_score":            cfg.min_cycle_score,
        "max_intersection_dist":      cfg.max_intersection_dist,
        "weight_distance":            cfg.weight_distance,
        "weight_score":               cfg.weight_score,
        "min_keypoints_for_grouping": cfg.min_keypoints_for_grouping,
        "min_kp_ratio":               cfg.min_kp_ratio,
        "min_fallback_score":         cfg.min_fallback_score,
        "kp_weights":                 cfg.kp_weights,
    }


def _run_reconstruction_step(matcher, reconstructor, annotations):
    """Passo comum a ambas as fontes: extrai, casa e reconstrói os esqueletos 3D."""
    skeletons_2d, ids_2d     = matcher.extract_skeletons_from_annotations(annotations)
    matched_persons          = matcher.match_skeletons(skeletons_2d, ids_2d, [1])
    reconstructed_skeletons  = reconstructor.reconstruct_all(matched_persons, annotations)

    skeletons_to_visualize = []
    for idx, skeleton_data in enumerate(reconstructed_skeletons):
        if not skeleton_data:
            continue

        skeletons_to_visualize.append({
            "id": idx + 1,
            "skeleton_3d": skeleton_data,
            "average_point": 0,
            "matche_2d": 0,
        })

    return skeletons_to_visualize


# ──────────────────────────────────────────────────────────────────────────
# Modo IS — entrada via Espaço Inteligente (broker AMQP)
# ──────────────────────────────────────────────────────────────────────────

def run_is(args):
    from is_utils.stream_handler import StreamHandler
    from is_utils.streamChannel import StreamChannel
    from is_wire.core import Message

    print("=== Skeleton Tracker 3D — fonte: IS (Espaço Inteligente) ===")
    print(f"  broker_uri           : {cfg.broker_uri}")
    print(f"  num_cameras (IS)     : {cfg.is_num_cameras}")
    print(f"  calib_path (IS)      : {cfg.is_calib_path}")
    print(f"  use_undistorted      : {cfg.use_undistorted}")
    print(f"  num_keypoints        : {cfg.num_keypoints}")
    print(f"  use_cycle_consistency: {cfg.use_cycle_consistency}")

    camera_files_path = calibration_files_path_is(cfg.is_calib_path, cfg.is_num_cameras)
    calib_files_data  = [np.load(f) for f in camera_files_path]

    matcher_params = _build_matcher_params()

    print("Inicializando módulos...")
    geometry              = FundamentalMatrices(camera_files_path, use_undistorted=cfg.use_undistorted)
    projection_matrices   = geometry.projection_matrices_all()
    fundamentals          = geometry.fundamental_matrices_all()
    all_calibs_parameters = geometry.get_all_calibs_parameters()

    stream        = StreamHandler(cfg.broker_uri, cfg.is_num_cameras, cfg.num_keypoints, all_calibs_parameters)
    channel       = StreamChannel(cfg.broker_uri)
    matcher       = SkeletonMatcher(fundamentals, matcher_params, cfg.is_num_cameras, cfg.num_keypoints)
    reconstructor = Reconstructor3D(projection_matrices, cfg.is_num_cameras, cfg.num_keypoints)
    visualizer    = Visualizer(all_calibs_parameters)

    print("Loop principal iniciado (IS).")
    while True:
        raw_messages = stream.get_latest_messages()
        if raw_messages is None:
            continue

        annotations = stream.prepare_input_data(raw_messages, calib_files_data)

        skeletons_to_visualize = _run_reconstruction_step(matcher, reconstructor, annotations)

        plot_img_bgr = visualizer.update(skeletons_to_visualize)

        rendered_msg = Message()
        rendered_msg.topic = "SkeletonDetector.3D"
        rendered_msg.pack(_to_image(plot_img_bgr))
        channel.publish(rendered_msg)

        skt_msg = Message()
        skt_msg.topic = "SkeletonDetector.3D.Annotations"
        skt_msg.body  = json.dumps(skeletons_to_visualize, default=numpy_to_list).encode("utf-8")
        channel.publish(skt_msg)


# ──────────────────────────────────────────────────────────────────────────
# Modo dataset — entrada via vídeos/imagens locais (VideoProcessor)
# ──────────────────────────────────────────────────────────────────────────

def run_dataset(args):
    from video_processor import VideoProcessor

    config_file = json.load(open('etc/config.json', 'r'))

    dataset_name = None
    if args.dataset_name is not None:
        dataset_name = args.dataset_name.lower()
        if dataset_name not in config_file.get("datasets", {}):
            print(f"Dataset '{dataset_name}' não encontrado. Usando default_initialization.")
            dataset_name = None

    if dataset_name is not None:
        ds_cfg = config_file["datasets"][dataset_name]
    else:
        ds_cfg = config_file["default_initialization"]

    calib_path       = ds_cfg['calib_path']
    data_path        = ds_cfg['data_path']
    num_cameras      = ds_cfg['num_cameras']
    data_type        = ds_cfg['data_type']
    prefix_name      = ds_cfg['prefix_name']
    file_extension   = ds_cfg['file_extension']
    use_undistorted  = ds_cfg['use_undistorted']
    apply_undistort  = ds_cfg['apply_undistort']
    filename_pattern = ds_cfg.get('filename_pattern', '{prefix}{index}{ext}')
    camera_ids_cfg   = ds_cfg.get('camera_ids', None)
    start_frame      = ds_cfg.get('start_frame', 0)
    total_frames     = ds_cfg.get('total_frames', 0)
    num_keypoints    = config_file.get('keypoint_settings', {}).get('num_keypoints', cfg.num_keypoints)

    camera_ids = (_parse_camera_ids(args.cameras, num_cameras)
                  if args.cameras else camera_ids_cfg)
    if camera_ids is None:
        camera_ids = list(range(num_cameras))
    num_cameras_used = len(camera_ids)

    print("=== Skeleton Tracker 3D — fonte: Dataset ===")
    print(f"  dataset              : {dataset_name or '(default_initialization)'}")
    print(f"  calib_path           : {calib_path}")
    print(f"  data_path            : {data_path}")
    print(f"  camera_ids           : {camera_ids}")
    print(f"  data_type            : {data_type}")
    print(f"  use_undistorted      : {use_undistorted}")
    print(f"  use_cycle_consistency: {cfg.use_cycle_consistency}")

    cameras_files = calibration_files_path_dataset(calib_path, camera_ids)
    geometry      = FundamentalMatrices(cameras_files, use_undistorted=use_undistorted)

    # all_calibs_parameters indexado por índice local (0-based, ordem de camera_ids)
    all_calibs_local     = geometry.get_all_calibs_parameters()
    projection_matrices  = geometry.projection_matrices_all()
    fundamentals         = geometry.fundamental_matrices_all()

    # Remapeia para cam_id original — necessário para VideoProcessor
    all_calibs_by_cam_id = {cam_id: all_calibs_local[i] for i, cam_id in enumerate(camera_ids)}

    yolo_model_name = config_file['yolo_model']['model_path']

    video_processor = VideoProcessor(
        all_calibs_parameters=all_calibs_by_cam_id,
        num_cameras=num_cameras_used,
        data_type=data_type,
        prefix_name=prefix_name,
        file_extension=file_extension,
        apply_undistort=apply_undistort,
        yolo_model_path=yolo_model_name,
        data_path=data_path,
        start_frame=start_frame,
        filename_pattern=filename_pattern,
        camera_ids=camera_ids,
    )

    if data_type == "images":
        total_frames = video_processor.num_frames()
        print(f"Total de frames a processar: {total_frames}")
        if total_frames == 0:
            print("Nenhum frame encontrado. Verifique o data_path no config.")
            return

    matcher_params = _build_matcher_params()
    matcher        = SkeletonMatcher(fundamentals, matcher_params, num_cameras_used, num_keypoints)
    reconstructor  = Reconstructor3D(projection_matrices, num_cameras_used, num_keypoints)
    visualizer     = Visualizer(all_calibs_local)

    print("Loop principal iniciado (dataset).")
    frame_idx = start_frame
    while (frame_idx < total_frames) if (data_type == "images") else True:
        frames, annotations, frame_paths = video_processor.process_next_frame()
        if annotations is None:
            print("Fim dos dados (vídeo/imagens esgotados).")
            break

        skeletons_to_visualize = _run_reconstruction_step(matcher, reconstructor, annotations)

        plot_img_bgr = visualizer.update(skeletons_to_visualize)

        grid = create_adaptive_camera_grid(
            frames,
            cell_height=360,
            cell_width=288,
            add_labels=True,
        )

        h_grid = grid.shape[0]
        h_map, w_map = plot_img_bgr.shape[:2]
        if h_map != h_grid:
            scale = h_grid / h_map
            plot_img_bgr = cv2.resize(
                plot_img_bgr,
                (int(w_map * scale), h_grid),
                interpolation=cv2.INTER_NEAREST,
            )

        imgcombined = np.hstack([grid, plot_img_bgr])
        cv2.imshow("Multi-view Skeleton Matching - Dataset", imgcombined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    video_processor.release_resources()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Skeleton Tracker 3D — IS ou Dataset")
    parser.add_argument('--source', type=str, choices=['is', 'dataset'], required=True,
                         help="Fonte de entrada: 'is' (Espaço Inteligente / broker) ou 'dataset' (vídeos/imagens locais).")
    parser.add_argument('--dataset_name', type=str, default=None,
                         help="[--source dataset] Nome do dataset (ex.: campus, shelf). Ignorado em --source is.")
    parser.add_argument('--cameras', type=str, default=None,
                         help="[--source dataset] Câmeras a usar: '0,1,3' | '0-4' | '0,2-4'. Ignorado em --source is.")
    args = parser.parse_args()

    if args.source == 'is':
        run_is(args)
    else:
        run_dataset(args)


if __name__ == "__main__":
    main()