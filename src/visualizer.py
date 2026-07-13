import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class Visualizer:
    """Visualizador 3D de esqueletos usando Matplotlib (backend Agg, sem display)."""

    SKELETON_CONNECTIONS = [
        (16,  14),  
        (14,  12),  
        (17,  15),  
        (15,  13),  
        (12,  13),  
        (6,  12),   
        (7,  13),   
        (6,  7),    
        (6,  8),    
        (8,  10),   
        (7,  9),    
        (9,  11),   
        (2,  3),    
        (1,  2),
        (1,  3),
        (2,  4),    
        (3,  5),    
    ]

    # Paleta de cores por pessoa (matplotlib color strings)
    PERSON_COLORS = [
        '#A855F7',  # violeta
        '#22D3EE',  # cyan
        '#4ADE80',  # verde
        '#FB923C',  # laranja
        '#F472B6',  # rosa
        '#FACC15',  # amarelo
    ]

    def __init__(self, all_calibs_parameters, size=(800, 600)):
        """
        Args:
            all_calibs_parameters: dict de parâmetros de calibração por câmera.
            size: tamanho da imagem de saída (largura, altura) em pixels.
        """
        self.all_calibs_parameters = all_calibs_parameters
        self.dpi = 100
        w_in = size[0] / self.dpi
        h_in = size[1] / self.dpi

        self.fig = plt.figure(figsize=(w_in, h_in), dpi=self.dpi, )
        self.ax: Axes3D = self.fig.add_subplot(111, projection='3d')

        # Ângulo de visão inicial
        self.ax.view_init(elev=20, azim=-110)

        # Limites padrão (metros)
        self._xlim = (-4.5, 4.5)
        self._ylim = (-4.5, 4.5)
        self._zlim = (0.0,  3.0)

        self._color_cache: dict = {}

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _get_color(self, person_id: int) -> str:
        if person_id not in self._color_cache:
            self._color_cache[person_id] = self.PERSON_COLORS[
                len(self._color_cache) % len(self.PERSON_COLORS)
            ]
        return self._color_cache[person_id]

    # Comprimento das setas dos eixos das câmeras (metros)
    CAMERA_AXIS_LEN = 0.3

    def _draw_cameras(self):
        """Plota os eixos locais (X=vermelho, Y=verde, Z=azul) de cada câmera."""
        axis_len = self.CAMERA_AXIS_LEN
        axis_colors = ('red', 'green', 'blue')   # X, Y, Z

        for cam_idx, params in self.all_calibs_parameters.items():
            rt = params['rt']          # (3, 4)
            R  = rt[:, :3]
            t  = rt[:, 3]
            cam_center = -R.T @ t      # centro óptico no mundo

            # Colunas de R.T são os eixos locais expressos no sistema mundo
            axes_world = R.T            # shape (3, 3) — cada coluna é um eixo

            for axis_idx, color in enumerate(axis_colors):
                direction = axes_world[:, axis_idx] * axis_len
                self.ax.quiver(
                    cam_center[0], cam_center[1], cam_center[2],
                    direction[0],  direction[1],  direction[2],
                    color=color, linewidth=1.5, arrow_length_ratio=0.3,
                )

            self.ax.text(
                cam_center[0], cam_center[1], cam_center[2] + axis_len + 0.05,
                f'Cam {cam_idx + 1}',
                color='black', fontsize=7, ha='center',
            )

    def _draw_skeleton(self, skeleton: dict, color: str):
        """Plota keypoints e conexões de um único esqueleto."""
        if not skeleton:
            return

        # Conexões
        for (src, dst) in self.SKELETON_CONNECTIONS:
            if src in skeleton and dst in skeleton:
                p1 = skeleton[src]
                p2 = skeleton[dst]
                self.ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color, linewidth=2.0, solid_capstyle='round',
                )

    def _configure_axes(self):
        """Estiliza os eixos."""
        ax = self.ax
        ax.set_xlim(*self._xlim)
        ax.set_ylim(*self._ylim)
        ax.set_zlim(*self._zlim)

        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_zlabel('Z (m)', fontsize=8)

        ax.tick_params(labelsize=6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False

        ax.grid(True, linewidth=0.5)

    def _fig_to_bgr(self) -> np.ndarray:
        """Converte a figura Matplotlib em array BGR (OpenCV)."""
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        rgba = np.asarray(buf, dtype=np.uint8)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def update(self, skeletons_to_visualize: list) -> np.ndarray:
        """
        Atualiza a visualização e retorna imagem BGR (numpy array).

        Args:
            skeletons_to_visualize: lista de dicts com chaves:
                - 'id'          : int
                - 'skeleton_3d' : dict {kp_id: [x, y, z]}

        Returns:
            np.ndarray BGR, shape (H, W, 3).
        """
        self.ax.clear()
        self._configure_axes()
        self._draw_cameras()

        num_people = 0
        for person in skeletons_to_visualize:
            skeleton = person.get('skeleton_3d', {})
            if not skeleton:
                continue
            color = "#A855F7" 
            self._draw_skeleton(skeleton, color)
            num_people += 1

        # Legenda discreta
        self.ax.set_title(
            f'Pessoas detectadas: {num_people}',
            color='white', fontsize=9, pad=4,
        )

        return self._fig_to_bgr()