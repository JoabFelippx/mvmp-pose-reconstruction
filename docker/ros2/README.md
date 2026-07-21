# Skeleton ROS 2 — Pipeline de Reconstrução 3D

Container Docker com ROS 2 rodando o pipeline de detecção/triangulação de esqueletos
(`skeleton_publisher`) e visualização em tempo real via RViz2.

## Pré-requisitos (uma vez por máquina)

- Docker + Docker Compose instalados
- Driver NVIDIA + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) configurado (necessário para aceleração CUDA do YOLO)
- X11 rodando no host (padrão em qualquer distro Linux com ambiente gráfico)

Verifique se a GPU está visível para o Docker antes de continuar:

```bash
nvidia-smi
```

Se esse comando falhar no host, resolva isso antes — sem GPU visível, o pipeline
cai para CPU (mais lento) ou falha na inicialização do YOLO.

## Rodando o container

**1. Autorize o Docker a desenhar na sua tela** (necessário 1x por sessão de login):

```bash
xhost +local:docker
```

**2. Suba o container:**

```bash
sudo docker compose -f docker/ros2/docker-compose.yaml up --build
```

Isso builda a imagem (se necessário) e sobe o `skeleton_publisher` + RViz2 já
configurado no tópico `/skeletons_3d`, com `Fixed Frame = world`.

Para rodar com um dataset específico ou outra taxa de publicação, edite os
parâmetros no `docker-compose.yaml` ou passe via launch:

```bash
ros2 launch skeleton_ros2 skeleton_pipeline.launch.py dataset_name:=shelf publish_rate_hz:=10.0
```

## Encerrando

`Ctrl+C` no terminal onde o `docker compose up` está rodando. Se quiser revogar
a permissão de X11 depois:

```bash
xhost -local:docker
```
