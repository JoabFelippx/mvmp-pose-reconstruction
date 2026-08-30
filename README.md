# Multi-view Multi-person 3D Pose Reconstruction

Pipeline para reconstrução 3D de poses humanas em cenários **multi-câmera** e **multi-pessoa**.

O sistema recebe poses 2D detectadas em diferentes câmeras, calcula a compatibilidade geométrica entre os esqueletos, refina as afinidades utilizando suporte de ciclo entre múltiplas vistas, realiza o matching entre câmeras e reconstrói os keypoints em 3D por triangulação DLT/SVD.

O pipeline suporta duas fontes de entrada:

- **`dataset`** — vídeos ou sequências de imagens locais, incluindo Campus e Shelf;
- **`is`** — streaming em tempo real via Espaço Inteligente (IS), consumindo mensagens de um broker AMQP.

Também é possível executar a detecção 2D separadamente e reutilizar os resultados salvos em JSON. Isso é útil para testar alterações no matching e na reconstrução sem executar a YOLO novamente.

![](figures/image1.png)

---

## Sumário

- [Pipeline](#pipeline)
- [Principais funcionalidades](#principais-funcionalidades)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Datasets e calibrações](#datasets-e-calibrações)
- [Configuração](#configuração)
- [Extração offline das poses 2D](#extração-offline-das-poses-2d)
- [Como rodar](#como-rodar)
  - [Dataset com YOLO em tempo de execução](#dataset-com-yolo-em-tempo-de-execução)
  - [Dataset com detecções 2D pré-computadas](#dataset-com-detecções-2d-pré-computadas)
  - [Modo IS](#modo-is)
- [Matching multi-câmera](#matching-multi-câmera)
- [Reconstrução e erro de reprojeção](#reconstrução-e-erro-de-reprojeção)
- [Visualização 3D](#visualização-3d)
- [Otimizações de desempenho](#otimizações-de-desempenho)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Problemas comuns](#problemas-comuns)

---

## Pipeline

O fluxo principal do sistema é:

```text
Imagens multi-câmera
        │
        ▼
Detecção de poses 2D
(YOLO Pose ou JSON pré-computado)
        │
        ▼
Afinidade geométrica
(geometria epipolar / erro de Sampson)
        │
        ▼
Suporte de ciclo multi-view
        │
        ▼
Matching pairwise
(Hungarian)
        │
        ▼
Agrupamento global
(uma detecção por câmera)
        │
        ▼
Triangulação
(DLT/SVD)
        │
        ▼
Esqueletos 3D
        │
        ├── Métricas de reprojeção
        │
        └── Visualização Vispy/OpenGL
```

As matrizes fundamentais e as matrizes de projeção são obtidas a partir das calibrações das câmeras.

Quando `use_undistorted=true`, o pipeline utiliza `nK`, correspondente à matriz intrínseca das imagens sem distorção.

---

## Principais funcionalidades

- detecção de poses 2D com YOLO Pose;
- suporte a detecções 2D pré-computadas;
- leitura de Campus e Shelf;
- suporte ao Espaço Inteligente via AMQP;
- geometria epipolar entre todos os pares de câmeras;
- afinidade baseada no erro de Sampson;
- refinamento das afinidades por suporte de ciclo;
- matching pairwise com algoritmo Húngaro;
- agrupamento global com restrição de uma detecção por câmera;
- reconstrução 3D por DLT/SVD;
- erro de reprojeção;
- métricas de reprojeção por câmera e por joint;
- visualização 3D com Vispy/OpenGL;
- câmera 3D fixa ou automática;
- processamento offline da YOLO com I/O paralelo e inferência em batch.

---

## Requisitos

- Python 3.11+
- NumPy
- OpenCV
- SciPy
- Ultralytics
- Vispy
- PyQt6
- GPU NVIDIA com CUDA recomendada para a detecção YOLO
- para o modo `is`: acesso a um broker AMQP compatível com o Espaço Inteligente

> O matching, a triangulação e a leitura de detecções pré-computadas não exigem executar a YOLO novamente.

---

## Instalação

Clone o repositório:

```bash
git clone https://github.com/JoabFelippx/mvmp-pose-reconstruction.git
cd mvmp-pose-reconstruction
```

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```


Caso Vispy/PyQt6 ainda não estejam no `requirements.txt`:

```bash
pip install vispy PyQt6
```

O modelo YOLO não precisa ser armazenado no GitHub. Coloque o arquivo localmente em `models/` ou informe outro caminho na configuração.

---

## Datasets e calibrações

As calibrações (arquivos `.npz`) — tanto do modo IS (`calibrations/`) quanto dos datasets Campus e Shelf (`datasets/Campus_Seq1/`, `datasets/Shelf_Seq1/`) — **já estão incluídas neste repositório**, já convertidas para o formato esperado pelo pipeline (`K`, `dist`, `rt`, opcionalmente `nK`/`roi`).
 
O que **não está neste repositório** são os vídeos/frames dos datasets (arquivos grandes demais para o GitHub). Eles foram retirados do paper:
 
> **Chen, L., Ai, H., Chen, R., Zhuang, Z., & Liu, S. (2020).** *Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS.* CVPR 2020.
> Repositório oficial: [longcw/crossview_3d_pose_tracking](https://github.com/longcw/crossview_3d_pose_tracking)
 
Nesse repositório os próprios autores disponibilizam um link único do [Google Drive](https://drive.google.com/drive/folders/1LJGcP2v0aQDmetnCzO2PiRP1v4jU6sFC?usp=drive_link) para baixar todos os datasets (Campus, Shelf e StoreLayout2) de uma vez. Basta seguir o link do repositório oficial acima, baixar as pastas `Campus_Seq1` e `Shelf_Seq1`, e copiar apenas a pasta `frames/` de cada uma para dentro de `datasets/Campus_Seq1/` e `datasets/Shelf_Seq1/` deste projeto (as calibrações `.npz` já estarão aqui, não é necessário baixá-las de novo nem convertê-las).

Estrutura esperada localmente após o download:

```text
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

---

## Configuração

O arquivo:

```text
etc/config.json
```

centraliza a configuração do sistema.

As seções principais são:

| Seção | Função |
|---|---|
| `default_initialization` | configuração padrão do modo dataset |
| `datasets.<nome>` | configuração específica de Campus, Shelf etc. |
| `is_settings` | broker, calibração e número de câmeras do modo IS |
| `matcher_parameters` | parâmetros do matching |
| `keypoint_settings` | quantidade e estrutura dos keypoints |
| `yolo_model` | configuração do detector YOLO |

A opção de suporte de ciclo pode ser controlada por:

```json
"use_cycle_consistency": true
```

Com ciclo ativado:

```text
afinidade epipolar
        ↓
suporte de ciclo
        ↓
Hungarian
```

Com ciclo desativado:

```text
afinidade epipolar
        ↓
Hungarian
```

Isso permite comparar experimentalmente o impacto do refinamento por ciclo.

---

## Extração offline das poses 2D

Para experimentos de matching é recomendável extrair as poses 2D uma única vez e salvar os resultados em JSON.

O script otimizado utiliza:

- leitura de imagens em múltiplas threads;
- undistortion paralela;
- mapas de undistortion calculados apenas uma vez por câmera;
- inferência YOLO em batch;
- processamento das câmeras de forma intercalada;
- salvamento periódico dos JSONs.

### Campus

Linux:

```bash
python src/extract_2d_yolo_threaded.py \
  --model yolo26m-pose.pt \
  --frames datasets/Campus_Seq1/frames \
  --calib datasets/Campus_Seq1/calib_cameras_campus \
  --output datasets/Campus_Seq1/detections_2d \
  --cameras 0,1,2 \
  --device 0 \
  --workers 6 \
  --batch-size 4 \
  --imgsz 1280 \
  --conf 0.20 \
  --quantize 16
```

### Shelf

```bash
python src/extract_2d_yolo_threaded.py \
  --model yolo26m-pose.pt \
  --frames datasets/Shelf_Seq1/frames \
  --calib datasets/Shelf_Seq1/calib_cameras_shelf \
  --output datasets/Shelf_Seq1/detections_2d \
  --cameras 0,1,2,3,4 \
  --device 0 \
  --workers 6 \
  --batch-size 4 \
  --imgsz 1280 \
  --conf 0.20 \
  --quantize 16
```

### Parâmetros de desempenho

`--workers` controla o número de threads utilizadas para leitura e pré-processamento das imagens.

`--batch-size` controla quantas imagens são enviadas em conjunto para a inferência.

A inferência da YOLO não é executada simultaneamente por múltiplas threads sobre a mesma instância do modelo. As threads são utilizadas para I/O e pré-processamento, enquanto a GPU recebe batches de imagens.

Se faltar VRAM, reduza:

```text
--batch-size 2
```

Se houver VRAM disponível, valores maiores podem ser testados.

---

## Como rodar

### Dataset com YOLO em tempo de execução

Campus:

```bash
python src/skeleton_tracker_main.py \
    --source dataset \
    --dataset_name campus
```

Shelf:

```bash
python src/skeleton_tracker_main.py \
    --source dataset \
    --dataset_name shelf
```

Também é possível selecionar apenas algumas câmeras:

```bash
python src/skeleton_tracker_main.py \
    --source dataset \
    --dataset_name shelf \
    --cameras 0,1,3
```

ou:

```bash
python src/skeleton_tracker_main.py \
    --source dataset \
    --dataset_name shelf \
    --cameras 0-3
```

### Dataset com detecções 2D pré-computadas

Campus:

```bash
python src/skeleton_tracker_main.py \
  --source dataset \
  --dataset_name campus \
  --input_2d precomputed \
  --detections_2d datasets/Campus_Seq1/detections_2d
```

Shelf:

```bash
python src/skeleton_tracker_main.py \
  --source dataset \
  --dataset_name shelf \
  --input_2d precomputed \
  --detections_2d datasets/Shelf_Seq1/detections_2d
```

A utilização de poses pré-computadas é recomendada durante o desenvolvimento do matching porque elimina o custo da inferência YOLO em cada execução.

### Modo IS

```bash
python src/skeleton_tracker_main.py --source is
```

O modo IS utiliza os parâmetros definidos em `is_settings` ou nas variáveis de ambiente correspondentes.

Os resultados são publicados nos tópicos:

```text
SkeletonDetector.3D
SkeletonDetector.3D.Annotations
```

---

## Matching multi-câmera

O matcher atual utiliza uma estratégia composta por quatro etapas.

### 1. Afinidade epipolar

Para cada par de esqueletos de câmeras diferentes é calculado o erro de Sampson dos keypoints correspondentes.

O score final combina:

- qualidade geométrica;
- quantidade de keypoints geometricamente compatíveis.

Somente pares que ultrapassam `min_compatibility_score` são mantidos na matriz de afinidade.

### 2. Suporte de ciclo

Para um par de câmeras `i` e `j`, uma terceira câmera `k` pode fornecer evidência indireta de que duas detecções correspondem à mesma pessoa.

Para cada candidato:

```text
i → k → j
```

é calculado um suporte max-product.

A afinidade direta é então combinada com o melhor suporte fornecido pelas câmeras intermediárias.

Essa etapa pode ser habilitada ou desabilitada por `use_cycle_consistency`.

### 3. Hungarian pairwise

Depois do refinamento das afinidades, o algoritmo Húngaro resolve o matching bipartido de cada par de câmeras.

Para `N` câmeras, são processados todos os pares possíveis.

Por exemplo, com 5 câmeras:

```text
C(5,2) = 10 pares
```

### 4. Agrupamento global

Os matches pairwise são ordenados pelo score.

Um Union-Find/DSU constrói os grupos globais de forma gulosa, respeitando a restrição:

> uma pessoa não pode possuir duas detecções provenientes da mesma câmera.
---

## Reconstrução e erro de reprojeção

Depois do matching, os keypoints correspondentes são triangulados utilizando DLT/SVD.

Para avaliar geometricamente a reconstrução, cada ponto 3D é projetado novamente nas câmeras em que foi originalmente observado.

O erro de reprojeção é:

```text
erro = || ponto_reprojetado - ponto_observado ||₂
```

em pixels.

O pipeline calcula:

- `mean_px`;
- `median_px`;
- `rmse_px`;
- `p95_px`;
- `num_points`;
- métricas por câmera;
- métricas por joint.

Essas informações também podem ser exibidas no visualizador 3D.

---

## Visualização 3D

A visualização 3D utiliza **Vispy/OpenGL** em vez de Matplotlib.

Os objetos gráficos são criados uma única vez e seus buffers são atualizados entre os frames.

Isso reduz o custo da visualização contínua.

O viewer mostra:

- joints 3D;
- conexões do esqueleto;
- grid e eixos de referência;
- cores diferentes por pessoa;
- erro médio de reprojeção;
- RMSE;
- quantidade de observações utilizadas.

A câmera pode ser configurada como automática ou fixa.

O pipeline atualmente utiliza:

```python
SkeletonViewer3D(
    size=(900, 700),
    auto_camera=False,
)
```

---

## Otimizações de desempenho

As principais otimizações implementadas são:

1. **detecções 2D pré-computadas**

   A YOLO pode ser executada apenas uma vez e os JSONs reutilizados em experimentos posteriores.

2. **mapas de undistortion pré-calculados**

   `cv2.initUndistortRectifyMap()` é executado apenas uma vez por câmera.

3. **I/O paralelo**

   Leitura e undistortion podem ser processadas com `ThreadPoolExecutor`.

4. **inferência YOLO em batch**

   Várias imagens são enviadas simultaneamente para a GPU.

5. **visualização com Vispy**

   A renderização 3D utiliza OpenGL e objetos visuais persistentes.

6. **matcher simplificado**

   O código antigo de interseção de epilinhas/MST foi removido, mantendo apenas as etapas efetivamente utilizadas pelo pipeline atual.

---

## Estrutura do projeto

```text
src/
├── skeleton_tracker_main.py
│   # entry point: dataset ou IS
│
├── extract_2d_yolo_threaded.py
│   # extração offline de poses 2D
│   # I/O paralelo + inferência YOLO em batch
│
├── config.py
│   # configuração por JSON e variáveis de ambiente
│
├── fundamental_matrices.py
│   # matrizes fundamentais e matrizes de projeção
│
├── skeleton_matcher.py
│   # afinidade epipolar
│   # suporte de ciclo
│   # Hungarian
│   # agrupamento global
│
├── reconstructor_3d.py
│   # triangulação DLT/SVD
│   # métricas de reprojeção
│
├── skeletons.py
│   # YOLO Pose
│   # conversão para ObjectAnnotations
│
├── video_processor.py
│   # leitura de vídeos/imagens
│   # detecções 2D pré-computadas
│   # undistortion
│
├── visualizer.py
│   # renderização 3D com Vispy/OpenGL
│
├── utils.py
│   # grid de câmeras e funções auxiliares
│
└── is_utils/
    ├── stream_handler.py
    └── streamChannel.py
```

---

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
| `MATCHER_*` | parâmetros correspondentes do matcher | ver `config.py` |

---
