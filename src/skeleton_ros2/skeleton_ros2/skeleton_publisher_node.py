#!/usr/bin/env python3
"""
skeleton_publisher_node.py

Nó ROS 2 que:
  1. Inicializa o pipeline existente (câmeras, matcher, reconstructor)
     reaproveitando as funções já escritas em skeleton_tracker_main.py.
  2. A cada tick de um timer, processa o próximo frame do dataset/vídeo.
  3. Converte a lista de esqueletos 3D reconstruídos em um
     visualization_msgs/MarkerArray.
  4. Publica esse MarkerArray no tópico /skeletons_3d, para ser
     visualizado no RViz2.

Como rodar (depois de buildado):
    ros2 run skeleton_ros2 skeleton_publisher --ros-args -p dataset_name:=shelf
"""

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# ── Módulos do pipeline "puro" (não-ROS), disponíveis via PYTHONPATH ────────
# Graças a ENV PYTHONPATH="/workspace/src:..." no Dockerfile, estes imports
# funcionam mesmo sem esses arquivos serem parte do pacote ROS 2.
from config import cfg
from fundamental_matrices import FundamentalMatrices
from skeleton_matcher import SkeletonMatcher
from reconstructor_3d import Reconstructor3D
from video_processor import VideoProcessor
from visualizer import Visualizer  # reaproveitamos SKELETON_CONNECTIONS
from skeleton_tracker_main import (
    calibration_files_path_dataset,
    _build_matcher_params,
    _run_reconstruction_step,
    _parse_camera_ids,
)


# ─────────────────────────────────────────────────────────────────────────
# Paleta de cores por pessoa (RGBA, 0.0–1.0) — mesmo espírito do
# Visualizer.PERSON_COLORS, mas já no formato que o RViz2 espera.
# ─────────────────────────────────────────────────────────────────────────
PERSON_COLORS_RGBA = [
    (0.66, 0.33, 0.97, 1.0),  # violeta
    (0.13, 0.83, 0.93, 1.0),  # cyan
    (0.29, 0.87, 0.50, 1.0),  # verde
    (0.98, 0.57, 0.24, 1.0),  # laranja
    (0.96, 0.45, 0.71, 1.0),  # rosa
    (0.98, 0.80, 0.08, 1.0),  # amarelo
]


class SkeletonPublisherNode(Node):
    """Publica esqueletos 3D reconstruídos como MarkerArray para o RViz2."""

    def __init__(self):
        # 'skeleton_publisher' é o nome do nó dentro do grafo ROS 2
        # (aparece em `ros2 node list`).
        super().__init__('skeleton_publisher')

        # ── Parâmetros ROS 2 ─────────────────────────────────────────────
        # Parâmetros são a forma "certa" de configurar um nó ROS 2 sem
        # editar código — dá pra sobrescrever via CLI ou launch file.
        # Ex.: ros2 run skeleton_ros2 skeleton_publisher --ros-args -p dataset_name:=shelf
        self.declare_parameter('dataset_name', 'shelf')
        self.declare_parameter('cameras', '')          # ex.: '0,1,2,3'
        self.declare_parameter('frame_id', 'world')     # frame fixo do RViz2
        self.declare_parameter('publish_rate_hz', 10.0)

        dataset_name = self.get_parameter('dataset_name').value
        cameras_str  = self.get_parameter('cameras').value
        self.frame_id = self.get_parameter('frame_id').value
        publish_rate  = self.get_parameter('publish_rate_hz').value

        # ── QoS (Quality of Service) do publisher ───────────────────────
        # RELIABLE = garante entrega (como TCP); pra visualização em tempo
        # real, BEST_EFFORT também seria aceitável, mas RELIABLE é mais
        # simples de casar com a config default do RViz2.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.publisher_ = self.create_publisher(MarkerArray, '/skeletons_3d', qos)

        # ── Inicialização do pipeline (reaproveitando skeleton_tracker_main) ──
        self.get_logger().info(f'Inicializando pipeline para dataset="{dataset_name}"...')
        self._init_pipeline(dataset_name, cameras_str)

        # ── Timer: substitui o `while True:` do script original ─────────
        # Em ROS 2, nós não bloqueiam num loop manual; o executor chama
        # este callback periodicamente. period_seconds = 1 / hz.
        period_seconds = 1.0 / float(publish_rate)
        self.timer = self.create_timer(period_seconds, self._on_timer)

        # IDs de marker já publicados no frame anterior — usados para
        # "limpar" pessoas que desapareceram (ver _build_marker_array).
        self._last_marker_ids = set()

        self.get_logger().info('Nó pronto. Publicando em /skeletons_3d.')

    # ─────────────────────────────────────────────────────────────────────
    # Inicialização do pipeline (equivalente à primeira metade de run_dataset)
    # ─────────────────────────────────────────────────────────────────────
    def _init_pipeline(self, dataset_name: str, cameras_str: str):
        config_file = json.load(open('etc/config.json', 'r'))

        if dataset_name and dataset_name in config_file.get('datasets', {}):
            ds_cfg = config_file['datasets'][dataset_name]
        else:
            ds_cfg = config_file['default_initialization']

        calib_path       = ds_cfg['calib_path']
        data_path        = ds_cfg['data_path']
        num_cameras      = ds_cfg['num_cameras']
        data_type        = ds_cfg['data_type']
        prefix_name      = ds_cfg['prefix_name']
        file_extension   = ds_cfg['file_extension']
        apply_undistort  = ds_cfg['apply_undistort']
        filename_pattern = ds_cfg.get('filename_pattern', '{prefix}{index}{ext}')
        camera_ids_cfg   = ds_cfg.get('camera_ids', None)
        start_frame      = ds_cfg.get('start_frame', 0)
        num_keypoints    = config_file.get('keypoint_settings', {}).get(
            'num_keypoints', cfg.num_keypoints
        )

        camera_ids = (_parse_camera_ids(cameras_str, num_cameras)
                      if cameras_str else camera_ids_cfg)
        if camera_ids is None:
            camera_ids = list(range(num_cameras))
        num_cameras_used = len(camera_ids)

        cameras_files = calibration_files_path_dataset(calib_path, camera_ids)
        geometry = FundamentalMatrices(cameras_files, use_undistorted=ds_cfg['use_undistorted'])

        all_calibs_local    = geometry.get_all_calibs_parameters()
        projection_matrices = geometry.projection_matrices_all()
        fundamentals         = geometry.fundamental_matrices_all()
        all_calibs_by_cam_id = {cam_id: all_calibs_local[i] for i, cam_id in enumerate(camera_ids)}

        yolo_model_name = config_file['yolo_model']['model_path']

        self.video_processor = VideoProcessor(
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

        matcher_params = _build_matcher_params()
        self.matcher = SkeletonMatcher(fundamentals, matcher_params, num_cameras_used, num_keypoints)
        self.reconstructor = Reconstructor3D(projection_matrices, num_cameras_used, num_keypoints)

        # Reaproveita a lista de conexões do esqueleto já definida no
        # Visualizer — assim as duas visualizações (matplotlib e RViz2)
        # ficam sempre consistentes, sem duplicar a lista manualmente.
        self.skeleton_connections = Visualizer.SKELETON_CONNECTIONS

        self.total_frames = (self.video_processor.num_frames()
                              if data_type == 'images' else None)
        self.frame_idx = start_frame

    # ─────────────────────────────────────────────────────────────────────
    # Callback do timer — roda 1 frame por chamada
    # ─────────────────────────────────────────────────────────────────────
    def _on_timer(self):
        if self.total_frames is not None and self.frame_idx >= self.total_frames:
            self.get_logger().info('Fim do dataset. Parando o timer.')
            self.timer.cancel()
            return

        frames, annotations, _ = self.video_processor.process_next_frame()
        if annotations is None:
            self.get_logger().info('Fim dos dados (vídeo/imagens esgotados).')
            self.timer.cancel()
            return

        skeletons_to_visualize = _run_reconstruction_step(
            self.matcher, self.reconstructor, annotations
        )

        marker_array = self._build_marker_array(skeletons_to_visualize)
        self.publisher_.publish(marker_array)

        self.frame_idx += 1

    # ─────────────────────────────────────────────────────────────────────
    # Conversão: lista de esqueletos 3D → MarkerArray
    # ─────────────────────────────────────────────────────────────────────
    def _build_marker_array(self, skeletons_to_visualize: list) -> MarkerArray:
        """
        Para cada pessoa detectada, cria 2 Markers:
          - um POINTS (esferas nos keypoints)
          - um LINE_LIST (segmentos conectando os keypoints, via
            Visualizer.SKELETON_CONNECTIONS)

        IDs de marker são determinísticos por pessoa (id*2 e id*2+1), então
        o mesmo person_id sempre atualiza os mesmos Markers no RViz2 em vez
        de criar novos a cada frame.
        """
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        current_ids = set()

        for person in skeletons_to_visualize:
            person_id = person['id']
            skeleton = person.get('skeleton_3d', {})
            if not skeleton:
                continue

            color = PERSON_COLORS_RGBA[person_id % len(PERSON_COLORS_RGBA)]

            # ── Marker 1: keypoints como esferas (POINTS) ────────────────
            points_marker = Marker()
            points_marker.header.frame_id = self.frame_id
            points_marker.header.stamp = now
            points_marker.ns = 'skeleton_keypoints'
            points_marker.id = person_id * 2          # ID par → keypoints
            points_marker.type = Marker.POINTS
            points_marker.action = Marker.ADD
            points_marker.scale.x = 0.05               # diâmetro X (m)
            points_marker.scale.y = 0.05                # diâmetro Y (m)
            points_marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
            points_marker.pose.orientation.w = 1.0      # quaternion identidade

            for kpt_id, (x, y, z) in skeleton.items():
                points_marker.points.append(Point(x=float(x), y=float(y), z=float(z)))

            marker_array.markers.append(points_marker)
            current_ids.add((points_marker.ns, points_marker.id))

            # ── Marker 2: ossos como linhas (LINE_LIST) ──────────────────
            lines_marker = Marker()
            lines_marker.header.frame_id = self.frame_id
            lines_marker.header.stamp = now
            lines_marker.ns = 'skeleton_bones'
            lines_marker.id = person_id * 2 + 1         # ID ímpar → ossos
            lines_marker.type = Marker.LINE_LIST
            lines_marker.action = Marker.ADD
            lines_marker.scale.x = 0.02                  # espessura da linha (m)
            lines_marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
            lines_marker.pose.orientation.w = 1.0

            # LINE_LIST espera pares de pontos: [p1, p2, p3, p4, ...]
            # onde (p1,p2) é um segmento, (p3,p4) é o próximo, etc.
            for src, dst in self.skeleton_connections:
                if src in skeleton and dst in skeleton:
                    x1, y1, z1 = skeleton[src]
                    x2, y2, z2 = skeleton[dst]
                    lines_marker.points.append(Point(x=float(x1), y=float(y1), z=float(z1)))
                    lines_marker.points.append(Point(x=float(x2), y=float(y2), z=float(z2)))

            marker_array.markers.append(lines_marker)
            current_ids.add((lines_marker.ns, lines_marker.id))

        # ── Limpeza: remove markers de pessoas que saíram de cena ────────
        # Sem isso, se a "pessoa 3" some, o marker dela fica "congelado"
        # no RViz2 para sempre, pois nunca mais recebe um ADD com ação
        # de update. Publicamos um DELETE explícito para cada ID que
        # existia no frame anterior mas não existe mais neste.
        # stale_ids = self._last_marker_ids - current_ids
        # for stale_id in stale_ids:
        #     delete_marker = Marker()
        #     delete_marker.header.frame_id = self.frame_id
        #     delete_marker.header.stamp = now
        #     delete_marker.id = stale_id
        #     delete_marker.action = Marker.DELETE
        #     marker_array.markers.append(delete_marker)

        # self._last_marker_ids = current_ids
        stale_ids = self._last_marker_ids - current_ids
        for stale_ns, stale_id in stale_ids:
            delete_marker = Marker()
            delete_marker.header.frame_id = self.frame_id
            delete_marker.header.stamp = now
            delete_marker.ns = stale_ns          
            delete_marker.id = stale_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        self._last_marker_ids = current_ids

        return marker_array


def main(args=None):
    rclpy.init(args=args)
    node = SkeletonPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.video_processor.release_resources()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
