import cv2
import numpy as np

from vispy import app, scene
from vispy.scene import transforms

app.use_app("pyqt6")

PERSON_COLORS = [
    [0.8,  0.3,  1.0,  1.0],
    [0.0,  0.85, 1.0,  1.0],
    [0.3,  1.0,  0.4,  1.0],
    [1.0,  0.6,  0.1,  1.0],
    [1.0,  0.4,  0.7,  1.0],
    [1.0,  0.9,  0.1,  1.0],
    [0.4,  0.7,  1.0,  1.0],
    [1.0,  0.3,  0.3,  1.0],
]


# COCO-17
SKELETON_EDGES = [
    # cabeça
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),

    # tronco
    (6, 7),
    (6, 12),
    (7, 13),
    (12, 13),

    # braço esquerdo
    (6, 8),
    (8, 10),

    # braço direito
    (7, 9),
    (9, 11),

    # perna esquerda
    (12, 14),
    (14, 16),

    # perna direita
    (13, 15),
    (15, 17),
]


class SkeletonViewer3D:

    def __init__(
        self,
        size=(900, 700),
        auto_camera=True,
    ):

        self.width, self.height = size
        self.auto_camera = auto_camera

        # --------------------------------------------------
        # Canvas Vispy
        # --------------------------------------------------

        self.canvas = scene.SceneCanvas(
            keys=None,
            bgcolor=(0.95, 0.95, 0.95, 1.0),
            size=size,
            show=False,
        )

        self.view = self.canvas.central_widget.add_view()

        # --------------------------------------------------
        # Câmera 3D
        # --------------------------------------------------

        self.camera = scene.cameras.TurntableCamera(
            fov=70,
            elevation=25,
            azimuth=-60,
        )

        self.view.camera = self.camera
        self.view.camera.distance = 6

        # --------------------------------------------------
        # Cena
        # --------------------------------------------------

        self._setup_scene()

        # --------------------------------------------------
        # Visuals persistentes
        #
        # Não recriamos esses objetos em cada frame.
        # Apenas fazemos set_data().
        # --------------------------------------------------

        self.lines = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="segments",
            width=3,
            method="gl",
            parent=self.view.scene,
        )

        self.markers = scene.visuals.Markers(
            parent=self.view.scene
        )

        # Primeira configuração de câmera
        self._camera_initialized = False

    # ======================================================
    # CENA
    # ======================================================

    def _setup_scene(self):

        # Chão
        grid = scene.visuals.GridLines(
            color=(0.4, 0.4, 0.4, 0.35),
            parent=self.view.scene,
        )

        # Dependendo de como seu mundo está orientado,
        # pode ser interessante deixar o grid no plano XY.
        grid.transform = transforms.MatrixTransform()

        # Eixos XYZ
        axis = scene.visuals.XYZAxis(
            parent=self.view.scene
        )

        axis.transform = transforms.STTransform(
            scale=(0.5, 0.5, 0.5)
        )

    # ======================================================
    # KEYPOINTS
    # ======================================================

    @staticmethod
    def _parse_keypoints(kp_dict):

        pts = {}

        if kp_dict is None:
            return pts

        for kp_id, value in kp_dict.items():

            kp_id = int(kp_id)

            # numpy
            if isinstance(value, np.ndarray):

                value = np.asarray(value).reshape(-1)

                if len(value) >= 3:

                    xyz = value[:3].astype(float)

                    if np.all(np.isfinite(xyz)):
                        pts[kp_id] = xyz

            # lista / tuple
            elif isinstance(value, (list, tuple)):

                if len(value) >= 3:

                    xyz = np.asarray(
                        value[:3],
                        dtype=float
                    )

                    if np.all(np.isfinite(xyz)):
                        pts[kp_id] = xyz

            # dict
            elif isinstance(value, dict):

                if all(
                    k in value
                    for k in ("x", "y", "z")
                ):

                    xyz = np.array([
                        value["x"],
                        value["y"],
                        value["z"],
                    ], dtype=float)

                    if np.all(np.isfinite(xyz)):
                        pts[kp_id] = xyz

        return pts

    # ======================================================
    # NORMALIZAR INPUT
    # ======================================================

    @staticmethod
    def _normalize_input(skeletons_3d):
        """Normaliza os formatos aceitos pelo viewer sem perder metadados.

        O formato principal do pipeline é uma lista de pessoas contendo
        ``id``, ``skeleton_3d``, ``matche_2d`` e ``reprojection``.
        """

        normalized = []

        # Formato simples: {person_id: {kp_id: [X, Y, Z], ...}}
        if isinstance(skeletons_3d, dict):
            for person_idx, skeleton in skeletons_3d.items():
                normalized.append({
                    "person_idx": int(person_idx),
                    "id": int(person_idx) + 1,
                    "skeleton_3d": skeleton,
                    "matche_2d": None,
                    "reprojection": None,
                })
            return normalized

        # Formato usado pelo skeleton_tracker_main.py
        if isinstance(skeletons_3d, list):
            for person_idx, person in enumerate(skeletons_3d):
                if not isinstance(person, dict):
                    continue

                skeleton = person.get("skeleton_3d", person)

                normalized.append({
                    "person_idx": person_idx,
                    "id": int(person.get("id", person_idx + 1)),
                    "skeleton_3d": skeleton,
                    "matche_2d": person.get("matche_2d"),
                    "reprojection": person.get("reprojection"),
                })

        return normalized

    # ======================================================
    # CAMERA
    # ======================================================

    def _update_camera(self, points):

        if len(points) == 0:
            return

        pts = np.asarray(
            points,
            dtype=np.float32
        )

        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)

        center = (
            mins + maxs
        ) / 2.0

        extent = maxs - mins

        max_extent = float(
            np.max(extent)
        )

        if max_extent < 1e-3:
            max_extent = 1.0

        self.camera.center = tuple(center)

        # distância aproximada para enquadrar tudo
        self.camera.distance = (
            max_extent * 2.5
        )

    # ======================================================
    # UPDATE
    # ======================================================

    def update(self, skeletons_3d):

        persons = self._normalize_input(
            skeletons_3d
        )

        all_points = []
        all_point_colors = []

        line_segments = []
        line_colors = []

        # --------------------------------------------------
        # pessoas
        # --------------------------------------------------

        for person in persons:

            person_idx = int(
                person["person_idx"]
            )

            skeleton = person[
                "skeleton_3d"
            ]

            person_id = int(person["id"])

            color = np.asarray(
                PERSON_COLORS[
                    (person_id - 1) % len(PERSON_COLORS)
                ],
                dtype=np.float32,
            )

            pts_by_id = self._parse_keypoints(
                skeleton
            )

            # ------------------------------------------
            # joints
            # ------------------------------------------

            for point in pts_by_id.values():

                all_points.append(
                    point
                )

                all_point_colors.append(
                    color
                )

            # ------------------------------------------
            # bones
            # ------------------------------------------

            for kp_a, kp_b in SKELETON_EDGES:

                if (
                    kp_a not in pts_by_id
                    or
                    kp_b not in pts_by_id
                ):
                    continue

                line_segments.extend([
                    pts_by_id[kp_a],
                    pts_by_id[kp_b],
                ])

                line_colors.extend([
                    color,
                    color,
                ])

        # ==================================================
        # ATUALIZAR LINHAS
        # ==================================================

        if line_segments:

            line_segments = np.asarray(
                line_segments,
                dtype=np.float32
            )

            line_colors = np.asarray(
                line_colors,
                dtype=np.float32
            )

            self.lines.visible = True

            self.lines.set_data(
                pos=line_segments,
                color=line_colors,
                connect="segments",
                width=3,
            )

        else:

            # Evita buffers internos inconsistentes no Vispy quando
            # nao ha segmentos neste frame.
            self.lines.visible = False

        # ==================================================
        # ATUALIZAR JOINTS
        # ==================================================

        if all_points:

            points_array = np.asarray(
                all_points,
                dtype=np.float32
            )

            colors_array = np.asarray(
                all_point_colors,
                dtype=np.float32
            )

            self.markers.visible = True

            self.markers.set_data(
                points_array,
                face_color=colors_array,
                edge_color=(0, 0, 0, 1),
                edge_width=0.5,
                size=6,
                symbol="o",
            )

        else:

            # Evita buffers internos inconsistentes no Vispy quando
            # nao ha pontos neste frame.
            self.markers.visible = False

        # ==================================================
        # CAMERA
        # ==================================================

        if (
            self.auto_camera
            and all_points
        ):

            self._update_camera(
                all_points
            )

        # ==================================================
        # RENDER
        # ==================================================

        img = self.canvas.render()

        img = np.asarray(img)

        # Vispy retorna RGBA
        bgr = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2BGR
        )

        # ==================================================
        # ERRO DE REPROJEÇÃO
        # ==================================================
        # O Reconstructor3D já calcula as métricas em pixels.
        # O OpenCV é usado somente para desenhar o texto sobre
        # a imagem renderizada pelo Vispy, o que é barato.

        text_y = 30

        for person in persons:
            reprojection = person.get("reprojection")

            if not isinstance(reprojection, dict):
                continue

            mean_px = reprojection.get("mean_px")
            rmse_px = reprojection.get("rmse_px")
            num_points = reprojection.get("num_points", 0)

            if mean_px is None or not np.isfinite(mean_px):
                continue

            person_id = int(
                person.get("id", person["person_idx"] + 1)
            )

            if rmse_px is not None and np.isfinite(rmse_px):
                text = (
                    f"Pessoa {person_id} | "
                    f"Reproj: {float(mean_px):.2f} px | "
                    f"RMSE: {float(rmse_px):.2f} px | "
                    f"Pts: {int(num_points)}"
                )
            else:
                text = (
                    f"Pessoa {person_id} | "
                    f"Reproj: {float(mean_px):.2f} px | "
                    f"Pts: {int(num_points)}"
                )


            person_id = int(person["id"])

            rgba = np.asarray(
                PERSON_COLORS[
                    (person_id - 1) % len(PERSON_COLORS)
                ],
                dtype=np.float32,
            )
            
            text_color = (
                int(rgba[2] * 255),
                int(rgba[1] * 255),
                int(rgba[0] * 255),
            )

            # Contorno escuro para manter a leitura sobre o fundo.
            cv2.putText(
                bgr,
                text,
                (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                text,
                (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2,
                cv2.LINE_AA,
            )

            text_y += 26

        return bgr

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        self.canvas.close()