import glob
import os

import cv2
import numpy as np

from skeletons import SkeletonsDetector


class VideoProcessor:

    def __init__(
        self,
        all_calibs_parameters,
        num_cameras,
        data_type,
        prefix_name,
        file_extension,
        apply_undistort,
        yolo_model_path,
        data_path=None,
        loop_frames=False,
        start_frame=0,
        filename_pattern="{prefix}{index}{ext}",
        camera_ids=None,
    ) -> None:
        """
        Parameters
        ----------
        filename_pattern : str
            Template para montar o nome do arquivo de cada câmera.
            Variáveis disponíveis:
              {prefix}  → prefix_name do config
              {index}   → id da câmera (suporta format-spec, ex.: {index:02d})
              {ext}     → file_extension do config
            Exemplos de padrões:
              "{prefix}{index}{ext}"        →  camera_1.avi        (default)
              "{prefix}{index:02d}{ext}"    →  hd_00_00.mp4        (pano)
              "{prefix}{index}"             →  Camera0/            (campus – pasta de imagens)

        camera_ids : list[int] | None
            IDs reais das câmeras a usar. Se None, usa range(num_cameras).
            Permite pular câmeras: ex. [0, 1, 3] usa câmeras 0, 1 e 3.
        """
        self.all_calibs_parameters = all_calibs_parameters
        self.data_type = data_type
        self.prefix_name = prefix_name
        self.file_extension = file_extension
        self.apply_undistort = apply_undistort
        self.filename_pattern = filename_pattern
        self.data_path = data_path

        # Resolve lista de IDs de câmera
        if camera_ids is not None:
            self.camera_ids = list(camera_ids)
        else:
            self.camera_ids = list(range(num_cameras))

        # num_cameras reflete quantas câmeras estão realmente em uso
        self.num_cameras = len(self.camera_ids)

        self.loop_frames = loop_frames
        self.current_frame_index = start_frame
        self.skeletons_detector = SkeletonsDetector(yolo_model_path)

        self.video_captures = []
        self.image_paths = []

        # Mapas de undistortion pré-computados por câmera (local_index → (map1, map2))
        # Gerados aqui mesmo pois all_calibs_parameters já contém "h" e "w".
        self._undistort_maps: dict[int, tuple] = {}
        if self.apply_undistort:
            self._precompute_undistort_maps()

        if self.data_type == "videos":
            if self.data_path is None:
                raise ValueError("data_path must be provided when data_type is 'videos'")
            self.load_videos()
        if self.data_type == "images":
            if self.data_path is None:
                raise ValueError("data_path must be provided when data_type is 'images'")
            self.load_images()

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_filename(self, cam_id: int) -> str:
        """
        Monta o nome do arquivo/diretório para a câmera `cam_id` aplicando
        o filename_pattern.  Suporta format-specs Python, ex. {index:02d}.
        """
        class _Var:
            def __init__(self, value):
                self._v = value
            def __format__(self, spec):
                return format(self._v, spec)
            def __str__(self):
                return str(self._v)

        return self.filename_pattern.format(
            prefix=self.prefix_name,
            index=_Var(cam_id),
            ext=self.file_extension,
        )

    def _precompute_undistort_maps(self) -> None:
        """
        Pré-computa os mapas de undistortion para todas as câmeras usando
        as dimensões já presentes em all_calibs_parameters ("h" e "w").
        Chamado uma única vez no __init__ quando apply_undistort=True.
        """
        for local_index, cam_id in enumerate(self.camera_ids):
            cam_params = self.all_calibs_parameters[cam_id]
            K = cam_params["K"]
            nK = cam_params["nK"]
            dist_coeffs = cam_params["dist_coeffs"]
            image_width = cam_params["w"]
            image_height = cam_params["h"]

            map1, map2 = cv2.initUndistortRectifyMap(
                K, dist_coeffs, None, nK,
                (image_width, image_height), cv2.CV_16SC2
            )
            roi = cam_params["roi"]  # (x, y, w, h) — região válida após undistortion
            self._undistort_maps[local_index] = (map1, map2, roi)
            print(
                f"  [VideoProcessor] Mapas de undistortion criados para "
                f"cam_id={cam_id} ({image_width}x{image_height}), roi={roi}"
            )

    def _undistort_frame(self, frame: np.ndarray, local_index: int) -> np.ndarray:
        """
        Aplica undistortion no frame inteiro usando remap com os mapas
        pré-computados e recorta a região válida (roi) para remover bordas pretas.
        """
          # Sem ROI, retorna o frame inteiro (pode conter bordas pretas)
        map1, map2, roi = self._undistort_maps[local_index]
        if roi is None:
            return frame
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        x, y, w, h = roi
        return undistorted[0:h, 0:w]

    # ──────────────────────────────────────────────────────────────────────────
    # Carregamento de dados
    # ──────────────────────────────────────────────────────────────────────────

    def load_videos(self):
        for cam_id in self.camera_ids:
            filename = self._build_filename(cam_id)
            video_path = os.path.join(self.data_path, filename)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video: {video_path}")
            if self.current_frame_index > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
                actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if actual != self.current_frame_index:
                    print(
                        f"  [VideoProcessor] Aviso: solicitado frame {self.current_frame_index}, "
                        f"posicionado em {actual} (cam_id={cam_id})"
                    )
            self.video_captures.append(cap)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(
                f"  [VideoProcessor] cam_id={cam_id} → {video_path} "
                f"(start={self.current_frame_index}, total={total})"
            )

    def load_images(self):
        for cam_id in self.camera_ids:
            dir_name = self.filename_pattern.format(
                prefix=self.prefix_name,
                index=cam_id,
                ext="",
            )
            camera_dir = os.path.join(self.data_path, dir_name)
            paths = sorted(glob.glob(os.path.join(camera_dir, f"*{self.file_extension}")))
            if len(paths) == 0:
                raise FileNotFoundError(
                    f"No images found for camera {cam_id} in path: {camera_dir}"
                )
            self.image_paths.append(paths)
            print(f"  [VideoProcessor] cam_id={cam_id}: {len(paths)} imagens em {camera_dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # Leitura de frames
    # ──────────────────────────────────────────────────────────────────────────

    def num_frames(self):
        if not self.image_paths:
            return 0
        return min(len(p) for p in self.image_paths)

    def _load_single_frame(self, local_index: int, frame_index: int):
        """local_index é a posição dentro de self.image_paths (0-based)."""
        image_paths = self.image_paths[local_index]
        if frame_index >= len(image_paths):
            if self.loop_frames:
                frame_index = frame_index % len(image_paths)
            else:
                raise IndexError(
                    f"Frame {frame_index} out of range for camera "
                    f"(cam_id={self.camera_ids[local_index]}, "
                    f"total={len(image_paths)})"
                )
        img = cv2.imread(image_paths[frame_index])
        if img is None:
            raise IOError(f"Could not read image: {image_paths[frame_index]}")
        return img

    def process_next_frame(self):
        frames = []
        frame_paths = []  # caminhos dos arquivos (apenas para data_type == "images")

        if self.data_type == "videos":
            for cap in self.video_captures:
                ret, frame = cap.read()
                if not ret:
                    return None, None, None
                frames.append(frame)
                frame_paths.append(None)

        elif self.data_type == "images":
            for local_idx in range(self.num_cameras):
                idx = self.current_frame_index
                path = self.image_paths[local_idx][idx] if idx < len(self.image_paths[local_idx]) else None
                frame = self._load_single_frame(local_idx, idx)
                frames.append(frame)
                frame_paths.append(path)
            self.current_frame_index += 1

        if not frames:
            return None, None, None

        # Aplica undistortion nos frames antes da detecção.
        # cv2.remap com mapas pré-computados é significativamente mais rápido
        # do que undistortPoints aplicado ponto-a-ponto por frame.
        if self.apply_undistort:
            frames = [
                self._undistort_frame(frame, local_idx)
                for local_idx, frame in enumerate(frames)
            ]

        results_list = self.skeletons_detector.detect(frames)
        annotations = []
        for local_idx, results in enumerate(results_list):
            scores = results.keypoints.conf
            if scores is not None:
                scores = scores.cpu().numpy().astype("float32")
            else:
                scores = np.array([])

            obs = self.skeletons_detector.to_object_annotations(
                results, scores, frames[local_idx].shape
            )
            # Undistortion já foi aplicada nos frames; keypoints detectados
            # já estão no espaço corrigido — nenhum pós-processamento necessário.
            annotations.append(obs)

        return frames, annotations, frame_paths

    def release_resources(self):
        for cap in self.video_captures:
            cap.release()