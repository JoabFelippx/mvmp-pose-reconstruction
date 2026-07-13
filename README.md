# Multi-view Multi-person 3D Pose Reconstruction

Reconstrução 3D de poses humanas multi-câmera e multi-pessoa. O pipeline detecta esqueletos 2D por câmera, faz a correspondência entre vistas usando geometria epipolar (matrizes fundamentais) e reconstrói os pontos 3D por triangulação (DLT/SVD).

O sistema suporta duas fontes de entrada:

- **`dataset`** — vídeos ou sequências de imagens locais (ex.: CMU Panoptic, Campus, Shelf).
- **`is`** — streaming em tempo real via [Espaço Inteligente (IS)](https://github.com/Lab-SEA), consumindo mensagens de um broker AMQP.

---

## Sumário

- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Datasets e calibrações](#datasets-e-calibrações)
- [Configuração (`etc/config.json`)](#configuração-etcconfigjson)
- [Como rodar](#como-rodar)
  - [Modo dataset](#modo-dataset)
  - [Modo IS (broker)](#modo-is-broker)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Problemas comuns](#problemas-comuns)

---

## Requisitos

- Python 3.11+
- GPU NVIDIA com CUDA (usado pelo YOLO para detecção de poses via Ultralytics)
- Para o modo `is`: acesso a um broker AMQP (RabbitMQ) rodando o [Espaço Inteligente](https://github.com/Lab-SEA)

## Instalação

```bash
git clone <url-deste-repositorio>
cd skeleton_3D_matching

python3 -m venv env
source env/bin/activate

pip install -r requirements.txt
```

Baixe também o modelo YOLO de pose (ex.: `yolo26m-pose.pt`) e coloque em `models/`, ou ajuste o caminho em `etc/config.json` → `yolo_model.model_path`.

## Datasets e calibrações

As calibrações (arquivos `.npz`) — tanto do modo IS (`calibrations/`) quanto dos datasets Campus e Shelf (`datasets/Campus_Seq1/`, `datasets/Shelf_Seq1/`) — **já estão incluídas neste repositório**, já convertidas para o formato esperado pelo pipeline (`K`, `dist`, `rt`, opcionalmente `nK`/`roi`).
 
O que **não está neste repositório** são os vídeos/frames dos datasets (arquivos grandes demais para o GitHub). Eles foram retirados do paper:
 
> **Chen, L., Ai, H., Chen, R., Zhuang, Z., & Liu, S. (2020).** *Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS.* CVPR 2020.
> Repositório oficial: [longcw/crossview_3d_pose_tracking](https://github.com/longcw/crossview_3d_pose_tracking)
 
Nesse repositório os próprios autores disponibilizam um link único do [Google Drive](https://drive.google.com/drive/folders/1LJGcP2v0aQDmetnCzO2PiRP1v4jU6sFC?usp=drive_link) para baixar todos os datasets (Campus, Shelf e StoreLayout2) de uma vez. Basta seguir o link do repositório oficial acima, baixar as pastas `Campus_Seq1` e `Shelf_Seq1`, e copiar apenas a pasta `frames/` de cada uma para dentro de `datasets/Campus_Seq1/` e `datasets/Shelf_Seq1/` deste projeto (as calibrações `.npz` já estarão aqui, não é necessário baixá-las de novo nem convertê-las).

Estrutura esperada localmente após o download:

```
skeleton_3D_matching/
├── calibrations/
│   ├── calib_rt1.npz
│   ├── calib_rt2.npz
│   ├── calib_rt3.npz
│   └── calib_rt4.npz
├── datasets/
│   ├── Campus_Seq1/
│   │   ├── calib_cameras_campus{0,1,2}.npz
│   │   └── frames/
│   │       ├── Camera0/*.jpg
│   │       ├── Camera1/*.jpg
│   │       └── Camera2/*.jpg
│   └── Shelf_Seq1/
│       ├── calib_cameras_shelf{0..4}.npz
│       └── frames/
│           ├── Camera0/*.jpg
│           └── ...
├── models/
│   └── yolo26m-pose.pt
└── etc/
    └── config.json
```

> **Nota:** o formato de calibração do repositório original (`calibration.json`) é diferente do usado aqui (`.npz` com `K`, `dist`, `rt`, opcionalmente `nK`/`roi`). Se você baixar as calibrações originais do Campus/Shelf, será necessário convertê-las para `.npz` no formato esperado por este pipeline antes de usá-las — os arquivos de vídeo/frames podem ser usados diretamente.

Cada arquivo `.npz` de calibração deve conter, no mínimo: `K` (intrínsecos), `dist` (coeficientes de distorção), `rt` (extrínsecos 3×4), e opcionalmente `nK` (intrínsecos da imagem sem distorção) e `roi`.

## Configuração (`etc/config.json`)

O arquivo `etc/config.json` centraliza todos os parâmetros do pipeline. As seções principais:

| Seção | Para quê serve |
|---|---|
| `default_initialization` | Parâmetros padrão do **modo dataset** quando nenhum `--dataset_name` é passado (vídeos locais) |
| `datasets.<nome>` | Overrides por dataset nomeado (ex.: `campus`, `shelf`) |
| `is_settings` | Parâmetros exclusivos do **modo IS**: `broker_uri`, `calib_path`, `num_cameras` |
| `matcher_parameters` | Pesos e limiares do algoritmo de correspondência entre vistas |
| `keypoint_settings` | Número de keypoints, nomes e conexões do esqueleto |
| `yolo_model` | Caminho do modelo de pose e parâmetros de inferência |

Toda opção numérica/booleana também pode ser sobrescrita por variável de ambiente (prioridade: **env var > config.json > default do código**) — veja a tabela completa em [Variáveis de ambiente](#variáveis-de-ambiente).

⚠️ **Atenção:** `is_settings.calib_path` e `default_initialization.calib_path` são independentes. O modo IS usa exclusivamente `is_settings.calib_path` (ou a env var `IS_CALIB_PATH`) — ajustar apenas o `default_initialization` não afeta a execução via broker.

## Como rodar

### Modo dataset

Usando um dataset nomeado (ex.: `campus`, `shelf`, conforme definido em `etc/config.json`):

```bash
python src/skeleton_tracker_main.py --source dataset --dataset_name campus
```

Usando a configuração `default_initialization` (vídeos locais, sem `--dataset_name`):

```bash
python src/skeleton_tracker_main.py --source dataset
```

Selecionando um subconjunto de câmeras (índices 0-based, aceita faixas):

```bash
python src/skeleton_tracker_main.py --source dataset --dataset_name shelf --cameras 0,1,3
python src/skeleton_tracker_main.py --source dataset --dataset_name shelf --cameras 0-3
```

Uma janela do OpenCV abre mostrando o grid das câmeras lado a lado com a reconstrução 3D. Pressione `q` para encerrar.

### Modo IS (broker)

Requer um broker AMQP acessível e câmeras publicando detecções 2D no tópico `SkeletonDetector.<i>.Detection`.

```bash
python src/skeleton_tracker_main.py --source is
```

Os parâmetros (`broker_uri`, `calib_path`, `num_cameras`) vêm de `etc/config.json` → `is_settings`, ou das variáveis de ambiente `BROKER_URI`, `IS_CALIB_PATH`, `IS_NUM_CAMERAS`. O resultado é publicado de volta no broker nos tópicos `SkeletonDetector.3D` (imagem renderizada) e `SkeletonDetector.3D.Annotations` (JSON com os esqueletos 3D).

## Estrutura do projeto

```
src/
├── skeleton_tracker_main.py      # entry point (--source is | dataset)
├── config.py                     # carregamento de config (json + env vars)
├── fundamental_matrices.py       # matrizes fundamentais e de projeção por par de câmeras
├── skeleton_matcher.py           # correspondência de esqueletos entre vistas
├── reconstructor_3d.py           # triangulação DLT/SVD e reconstrução 3D
├── skeletons.py                  # detecção 2D (YOLO pose) e conversão para IS ObjectAnnotations
├── video_processor.py            # leitura de vídeos/imagens locais + undistortion
├── visualizer.py                 # renderização 3D (Matplotlib) do esqueleto reconstruído
├── utils.py                      # helpers de desenho e montagem de grid de câmeras
└── is_utils/
    ├── stream_handler.py         # consumo de mensagens do broker (threads por câmera)
    └── streamChannel.py          # wrapper de canal IS com descarte de mensagens antigas
```

## Variáveis de ambiente

| Variável | Sobrescreve | Default |
|---|---|---|
| `BROKER_URI` | `is_settings.broker_uri` | `amqp://guest:guest@localhost:5672` |
| `IS_CALIB_PATH` | `is_settings.calib_path` | `calibrations/calib_rt` |
| `IS_NUM_CAMERAS` | `is_settings.num_cameras` | `4` |
| `NUM_CAMERAS` | `default_initialization.num_cameras` | `4` |
| `CALIB_PATH` | `default_initialization.calib_path` | `calibrations/calib_rt` |
| `USE_UNDISTORTED` | `default_initialization.use_undistorted` | `true` |
| `APPLY_UNDISTORT` | `default_initialization.apply_undistort` | `true` |
| `NUM_KEYPOINTS` | `keypoint_settings.num_keypoints` | `18` |
| `MATCHER_*` | parâmetros correspondentes em `matcher_parameters` | ver `config.py` |

