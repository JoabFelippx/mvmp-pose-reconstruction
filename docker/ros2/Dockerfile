# ─────────────────────────────────────────────────────────────────
# Dockerfile — skeleton_3d_matching + ROS 2 Humble
# Base: ROS 2 Humble (Ubuntu 22.04) — LTS, boa compatibilidade Jetson
# ─────────────────────────────────────────────────────────────────
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── Dependências de sistema ─────────────────────────────────────────
# libgl1/libglib: necessárias pro OpenCV headless não quebrar em import
# python3-pip: pip do sistema (ROS Humble já vem com Python 3.10)
# ros-humble-rviz2: visualizador
# ros-humble-vision-msgs / rviz-visual-tools: úteis se quiser Marker helpers


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ros-humble-rviz2 \
    ros-humble-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Dependências Python do pipeline ─────────────────────────────────
# Copiamos requirements.txt primeiro para aproveitar cache de layer
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Fix: colcon/distutils do Humble não suporta setuptools >= 71 (API canonicalize_version mudou)
RUN pip3 install --no-cache-dir "setuptools<66"

# rclpy já vem com a imagem ros:humble-ros-base — não precisa pip install

# ── Código do projeto ────────────────────────────────────────────────
# Em dev, isso normalmente é sobrescrito por um bind-mount no
# docker-compose (ver serviço abaixo), então o COPY aqui garante que a
# imagem também funciona standalone (ex.: em produção/CI).
COPY src/ /workspace/src/
COPY etc/ /workspace/etc/
COPY calibrations/ /workspace/calibrations/

# Torna os módulos do pipeline (config.py, skeleton_matcher.py, etc.)
# importáveis de qualquer lugar, incluindo de dentro do pacote ROS 2.
ENV PYTHONPATH="/workspace/src:${PYTHONPATH}"

# ── Build do workspace ROS 2 ─────────────────────────────────────────
# Só compila o(s) pacote(s) ROS 2 que estiverem em src/ (ex.: skeleton_ros2).
# Pacotes .py "soltos" (config.py, etc.) são ignorados pelo colcon.
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

# ── Entrypoint ────────────────────────────────────────────────────────
# Faz source do ROS + do workspace automaticamente em todo `docker exec`/CMD
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /workspace/install/setup.bash" >> /root/.bashrc

COPY docker/ros2/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["ros2", "launch", "skeleton_ros2", "skeleton_pipeline.launch.py"]
