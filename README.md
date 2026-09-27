# Multi-view Multi-person 3D Pose Reconstruction

Pipeline para reconstrução 3D de poses humanas em cenários **multi-câmera** e **multi-pessoa**.

O sistema recebe poses 2D detectadas em diferentes câmeras, combina compatibilidade epipolar e distância no chão, aplica suporte de ciclo opcional, realiza o matching entre câmeras e reconstrói os keypoints em 3D por triangulação DLT/SVD. No modo dataset, um filtro de Kalman mantém os IDs das pessoas entre frames.

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
- [Rastreamento temporal](#rastreamento-temporal)
- [Exportação das reconstruções 3D](#exportação-das-reconstruções-3d)
- [Avaliação com Ground Truth](#avaliação-com-ground-truth)
- [Resultados experimentais](#resultados-experimentais)
- [Geração dos gráficos](#geração-dos-gráficos)
- [Visualização 3D](#visualização-3d)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Variáveis de ambiente](#variáveis-de-ambiente)

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
Suporte de ciclo multi-view (opcional)
        │
        ▼
Afinidade por distância no chão
(d0 específico do dataset)
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
        ▼
Rastreamento temporal no dataset
(Kalman em XY + Hungarian)
        │
        ├── Métricas de reprojeção
        ├── Exportação JSON
        │       │
        │       └── Avaliação com Ground Truth
        │               ├── PCP3D
        │               ├── Recall@500mm
        │               └── MPJPE
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
- afinidade por distância no chão, com `d0=1.25` no Campus e `d0=0.25` no Shelf;
- refinamento das afinidades por suporte de ciclo;
- matching pairwise com algoritmo Húngaro;
- agrupamento global com restrição de uma detecção por câmera;
- reconstrução 3D por DLT/SVD;
- rastreamento temporal no dataset com Kalman de velocidade constante em XY;
- erro de reprojeção;
- métricas de reprojeção por câmera e por joint;
- exportação das reconstruções 3D em JSON COCO17;
- avaliação em Campus e Shelf com PCP3D, Recall@500mm e MPJPE;
- comparação experimental com e sem cycle consistency;
- geração automática de gráficos a partir dos arquivos de métricas;
- visualização 3D com Vispy/OpenGL;
- câmera 3D fixa ou automática;
- processamento offline da YOLO com I/O paralelo e inferência em batch.

---

## Requisitos

- Python 3.11+
- NumPy
- OpenCV
- SciPy
- Matplotlib
- para inferência/extração YOLO: Ultralytics e PyTorch; o detector em tempo de execução utiliza `cuda:0`
- para visualização 3D: Vispy, PyQt6 e OpenGL desktop
- para o modo `is`: acesso a um broker AMQP compatível com o Espaço Inteligente

> Com `--input_2d precomputed`, o pipeline não importa Ultralytics/PyTorch, não carrega o modelo YOLO e não realiza o aquecimento da rede. Esse modo pode ser executado em CPU.

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


`requirements.txt` contém as dependências para reconstrução com poses pré-computadas, visualização e avaliação. Para executar a YOLO ou extrair novas detecções 2D, instale também as dependências opcionais:

```bash
pip install -r requirements-yolo.txt
```

Ultralytics instala o PyTorch como dependência. O modelo configurado para inferência é `models/yolo26x-pose.pt`; ajuste `yolo_model.model_path` se necessário. A configuração do modelo só é lida no modo YOLO.

---

## Datasets e calibrações

As calibrações do modo IS estão em `calibrations/`. O checkout também contém as calibrações do Campus em `datasets/Campus_Seq1/calib_cameras_campus/`. As calibrações e detecções 2D do Shelf devem ser fornecidas externamente.

O `etc/config.json` atual usa dados externos em `/mnt/datasets`:

| Dataset | Calibrações | Frames |
|---|---|---|
| Campus | `/mnt/datasets/Campus/Campus_Seq1/calib_cameras_campus` | `/mnt/datasets/Campus/Campus_Seq1/frames` |
| Shelf | `/mnt/datasets/Shelf/Shelf_Seq1/calib_cameras_shelf` | `/mnt/datasets/Shelf/Shelf_Seq1/frames` |

Altere `calib_path` e `data_path` de cada dataset se os arquivos estiverem em outro local. Mesmo com `--input_2d precomputed`, o pipeline ainda lê os frames e precisa das calibrações.

As anotações 3D utilizadas na avaliação também estão armazenadas em:

```text
datasets/Campus_Seq1/annotation_3d.json
datasets/Shelf_Seq1/annotation_3d.json
```

O que **não está neste repositório** são os vídeos/frames dos datasets (arquivos grandes demais para o GitHub). Eles foram retirados do paper:
 
> **Chen, L., Ai, H., Chen, R., Zhuang, Z., & Liu, S. (2020).** *Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS.* CVPR 2020.
> Repositório oficial: [longcw/crossview_3d_pose_tracking](https://github.com/longcw/crossview_3d_pose_tracking)
 
O download dos datasets está disponível no [Google Drive](https://drive.google.com/drive/folders/1LJGcP2v0aQDmetnCzO2PiRP1v4jU6sFC?usp=drive_link) indicado pelo repositório original. Organize os frames e as calibrações `.npz` nos caminhos definidos no `config.json`.

Estrutura esperada localmente após o download:

```text
/mnt/datasets/
├── Campus/Campus_Seq1/
│   ├── calib_cameras_campus/calib_rt{0,1,2}.npz
│   ├── detections_2d/camera_{0,1,2}.json
│   └── frames/Camera{0,1,2}/*.jpg
└── Shelf/Shelf_Seq1/
    ├── calib_cameras_shelf/calib_rt{0..4}.npz
    ├── detections_2d/camera_{0..4}.json
    └── frames/Camera{0..4}/*.jpg
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

Execute os scripts a partir da raiz do repositório, pois os caminhos de configuração são relativos a ela.

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
combinação com distância no chão
        ↓
Hungarian
```

Com ciclo desativado:

```text
afinidade epipolar
        ↓
combinação com distância no chão
        ↓
Hungarian
```

Isso permite comparar experimentalmente o impacto do refinamento por ciclo.

O parâmetro `datasets.<nome>.distance_d0` define o ponto em que a afinidade de distância vale `0.5`: Campus usa `1.25` e Shelf usa `0.25`. O valor acompanha a unidade das calibrações (metros nos datasets usados). Para outras fontes, o fallback é `matcher_parameters.distance_d0`, com default `1.25`; a variável `MATCHER_DISTANCE_D0` sobrescreve esse fallback. O valor específico do dataset tem prioridade.

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
  --frames /mnt/datasets/Campus/Campus_Seq1/frames \
  --calib /mnt/datasets/Campus/Campus_Seq1/calib_cameras_campus \
  --output /mnt/datasets/Campus/Campus_Seq1/detections_2d \
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
  --frames /mnt/datasets/Shelf/Shelf_Seq1/frames \
  --calib /mnt/datasets/Shelf/Shelf_Seq1/calib_cameras_shelf \
  --output /mnt/datasets/Shelf/Shelf_Seq1/detections_2d \
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
  --detections_2d /mnt/datasets/Campus/Campus_Seq1/detections_2d
```

Shelf:

```bash
python src/skeleton_tracker_main.py \
  --source dataset \
  --dataset_name shelf \
  --input_2d precomputed \
  --detections_2d /mnt/datasets/Shelf/Shelf_Seq1/detections_2d
```

A utilização de poses pré-computadas elimina o custo da inferência YOLO em cada execução. Basta instalar `requirements.txt`; Ultralytics, PyTorch, CUDA e os pesos da YOLO não são necessários nesse modo. Cada câmera selecionada precisa de um arquivo `camera_<id>.json` no diretório informado.

Use `--no_visualization` para exportar resultados sem carregar VisPy/PyQt6 nem criar um contexto OpenGL.

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

O matcher atual utiliza uma estratégia composta por cinco etapas.

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

### 3. Afinidade por distância no chão

O matcher projeta um ponto sob o quadril e os tornozelos detectados no plano `z=0` usando as calibrações. A distância entre os centros projetados de duas vistas é convertida em afinidade:

```text
A_distance = 1 / (1 + exp(20 * (distance - d0)))
A_final = sqrt(A_epipolar * A_distance)
```

O argumento da exponencial é limitado a `[-60, 60]` para estabilidade numérica. `A_epipolar` já inclui o suporte de ciclo quando ele está habilitado. Se uma projeção não estiver disponível, o matcher preserva a afinidade epipolar daquele par. Os campos `weight_distance` e `weight_score` não são usados nessa média geométrica.

| Dataset | `d0` |
|---|---:|
| Campus | 1.25 |
| Shelf | 0.25 |

### 4. Hungarian pairwise

Depois do refinamento das afinidades, o algoritmo Húngaro resolve o matching bipartido de cada par de câmeras.

Para `N` câmeras, são processados todos os pares possíveis.

Por exemplo, com 5 câmeras:

```text
C(5,2) = 10 pares
```

### 5. Agrupamento global

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

## Rastreamento temporal

No modo dataset, `SkeletonTracker3D` calcula o centro XY entre os quadris reconstruídos (IDs 12 e 13). Cada pessoa mantém um filtro de Kalman de velocidade constante com estado `[x, y, vx, vy]`.

O filtro prediz o centro no próximo timestamp, e o Hungarian associa as detecções aos tracks pela distância XY. Os defaults são `max_distance=0.30` e `max_missed=10`. Detecções sem associação recebem um novo ID; tracks são removidos ao superar o limite de frames perdidos. O filtro atualiza o centro usado na associação, enquanto os keypoints exportados continuam sendo os pontos triangulados.

Os timestamps vêm de `time.monotonic()` durante o processamento. Portanto, o intervalo usado pelo filtro é o tempo de execução entre frames, e não o intervalo original de captura do dataset.

---

## Exportação das reconstruções 3D

As reconstruções produzidas pelo pipeline podem ser armazenadas em JSON para avaliação posterior. Dessa forma, alterações no código de avaliação não exigem executar novamente detecção 2D, matching e triangulação.

O arquivo é salvo no formato **COCO17**, com um vetor de 17 posições por pessoa. Joints que não puderam ser reconstruídos são armazenados como `null`, evitando confundir ausência de informação com a coordenada `[0, 0, 0]`.

Exemplo simplificado:

```json
{
  "dataset": "campus",
  "keypoint_format": "COCO17",
  "coordinate_system": "world",
  "use_cycle_consistency": true,
  "frames": {
    "350": {
      "persons": [
        {
          "id": 1,
          "keypoints_3d": [
            [0.12, 1.03, 1.71],
            null
          ],
          "matched_2d": {
            "0": 1,
            "1": 0,
            "2": 2
          }
        }
      ]
    }
  }
}
```

O argumento `--cycle` permite forçar a execução com ou sem o refinamento por consistência de ciclo, independentemente do valor definido em `config.json`. Os exemplos de execução abaixo utilizam Linux/bash.

### Campus — com cycle consistency

Linux:

```bash
python src/skeleton_tracker_main.py \
    --source dataset \
    --dataset_name campus \
    --input_2d precomputed \
    --detections_2d /mnt/datasets/Campus/Campus_Seq1/detections_2d \
    --save_3d_json results/campus_3d.json \
    --save_every 100 \
    --no_visualization \
    --cycle on
```

### Campus — sem cycle consistency

```bash
python src/skeleton_tracker_main.py \
    --source dataset \
    --dataset_name campus \
    --input_2d precomputed \
    --detections_2d /mnt/datasets/Campus/Campus_Seq1/detections_2d \
    --save_3d_json results/campus_3d_no_cycle.json \
    --save_every 100 \
    --no_visualization \
    --cycle off
```

### Shelf — com cycle consistency

```bash
python src/skeleton_tracker_main.py \
    --source dataset \
    --dataset_name shelf \
    --input_2d precomputed \
    --detections_2d /mnt/datasets/Shelf/Shelf_Seq1/detections_2d \
    --save_3d_json results/shelf_3d.json \
    --save_every 100 \
    --no_visualization \
    --cycle on
```

### Shelf — sem cycle consistency

```bash
python src/skeleton_tracker_main.py \
    --source dataset \
    --dataset_name shelf \
    --input_2d precomputed \
    --detections_2d /mnt/datasets/Shelf/Shelf_Seq1/detections_2d \
    --save_3d_json results/shelf_3d_no_cycle.json \
    --save_every 100 \
    --no_visualization \
    --cycle off
```

A opção `--no_visualization` é recomendada para benchmarks completos, pois evita o custo de renderização do VisPy/OpenCV. `--save_every` controla a frequência dos checkpoints do JSON.

---

## Avaliação com Ground Truth

As reconstruções 3D dos datasets **Campus** e **Shelf** podem ser comparadas com as anotações 3D armazenadas nos arquivos `annotation_3d.json`.

O avaliador utilizado é:

```text
src/evaluate_shelf_campus.py
```

As predições do pipeline usam COCO17, enquanto as anotações de Campus/Shelf possuem 14 joints. O script realiza a conversão antes do cálculo das métricas.

### Métricas

A métrica principal é **PCP3D (Percentage of Correctly estimated Parts)**, com `alpha = 0.5`.

Também são calculadas métricas auxiliares:

- PCP3D por ator;
- PCP3D por grupo corporal;
- Recall@500mm;
- MPJPE da predição mais próxima do Ground Truth.

Os grupos corporais usados no relatório são:

- Head;
- Torso;
- Upper arms;
- Lower arms;
- Upper legs;
- Lower legs.

### Frames avaliados

O protocolo utilizado considera:

- **Campus:** frames `350–470` e `650–750`, totalizando 222 frames;
- **Shelf:** frames `300–600`, totalizando 301 frames.

### Campus

Com cycle consistency:

```bash
python src/evaluate_shelf_campus.py \
    --dataset campus \
    --predictions results/campus_3d.json \
    --gt datasets/Campus_Seq1/annotation_3d.json \
    --gt-unit cm \
    --pred-unit m \
    --output results/campus_metrics.json
```

Sem cycle consistency:

```bash
python src/evaluate_shelf_campus.py \
    --dataset campus \
    --predictions results/campus_3d_no_cycle.json \
    --gt datasets/Campus_Seq1/annotation_3d.json \
    --gt-unit cm \
    --pred-unit m \
    --output results/campus_metrics_no_cycle.json
```

### Shelf

Com cycle consistency:

```bash
python src/evaluate_shelf_campus.py \
    --dataset shelf \
    --predictions results/shelf_3d.json \
    --gt datasets/Shelf_Seq1/annotation_3d.json \
    --gt-unit cm \
    --pred-unit m \
    --output results/shelf_metrics.json
```

Sem cycle consistency:

```bash
python src/evaluate_shelf_campus.py \
    --dataset shelf \
    --predictions results/shelf_3d_no_cycle.json \
    --gt datasets/Shelf_Seq1/annotation_3d.json \
    --gt-unit cm \
    --pred-unit m \
    --output results/shelf_metrics_no_cycle.json
```

As predições reconstruídas estão em metros e são convertidas para milímetros durante a avaliação. As anotações `annotation_3d.json` estão em centímetros e também são convertidas para milímetros.

---

## Resultados experimentais

Os experimentos abaixo utilizam as mesmas detecções 2D e o mesmo método de triangulação. A variável experimental é a utilização do refinamento das afinidades por suporte de ciclo antes do matching Hungarian.

### Resultados gerais

| Dataset | Cycle support | PCP3D (%) | Recall@500mm (%) | MPJPE (mm) |
|---|---:|---:|---:|---:|
| Campus | Não | **95.28** | **99.73** | **88.30** |
| Campus | Sim | 92.51 | 96.54 | 154.92 |
| Shelf | Não | 93.18 | 99.02 | 80.60 |
| Shelf | Sim | **96.02** | **99.80** | **74.42** |

No **Shelf**, o suporte de ciclo aumentou o PCP3D de `93.18%` para `96.02%` e reduziu o MPJPE de `80.60 mm` para `74.42 mm`.

No **Campus**, entretanto, o mesmo refinamento reduziu o PCP3D de `95.28%` para `92.51%` e aumentou o MPJPE de `88.30 mm` para `154.92 mm`.

Esse comportamento mostra que o impacto do suporte de ciclo depende da qualidade das afinidades disponíveis entre as diferentes vistas e motiva a investigação de estratégias mais robustas para incorporar a informação de ciclo ao matching.

> **Nota:** esses valores são resultados dos experimentos já armazenados no repositório. As mudanças na afinidade por distância e no rastreamento com Kalman exigem uma nova execução para medir o desempenho do pipeline atual.

### Visualização dos resultados

#### PCP3D médio

![PCP3D médio](results/plots/01_avg_pcp3d.png)

#### MPJPE

![MPJPE](results/plots/03_mpjpe.png)

#### PCP3D por grupo corporal

![PCP3D por grupo corporal](results/plots/05_bone_groups.png)

Os demais gráficos estão disponíveis em `results/plots/`.

---

## Geração dos gráficos

Os gráficos podem ser regenerados a partir dos arquivos `*_metrics.json` utilizando `src/plot_metrics.py`.

Linux:

```bash
python src/plot_metrics.py \
    --metrics \
    results/campus_metrics_no_cycle.json \
    results/campus_metrics.json \
    results/shelf_metrics_no_cycle.json \
    results/shelf_metrics.json \
    --labels \
    "Campus sem cycle" \
    "Campus com cycle" \
    "Shelf sem cycle" \
    "Shelf com cycle" \
    --output-dir results/plots
```

São gerados:

```text
results/plots/
├── 01_avg_pcp3d.png
├── 02_recall.png
├── 03_mpjpe.png
├── 04_actor_pcp.png
├── 05_bone_groups.png
└── summary.txt
```

## Visualização 3D

O visualizador usa VisPy com PyQt6 e renderiza em uma superfície fora da tela com contexto OpenGL desktop. O contexto próprio evita a seleção de OpenGL ES pelo Qt no Wayland. O resultado é exibido junto ao grid de câmeras pelo OpenCV.

Para executar sem visualização, use `--no_visualization` no modo dataset. Nesse caso, VisPy/PyQt6 são carregados apenas se a visualização for solicitada.

---

## Estrutura do projeto

```text
src/
├── skeleton_tracker_main.py
│   # entry point: dataset ou IS
│   # execução e exportação das reconstruções 3D
│
├── prediction_exporter.py
│   # serialização das reconstruções em JSON COCO17
│
├── evaluate_shelf_campus.py
│   # avaliação PCP3D, Recall@500mm e MPJPE
│
├── plot_metrics.py
│   # geração dos gráficos dos experimentos
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
│   # afinidade por distância no chão
│   # Hungarian
│   # agrupamento global
│
├── reconstructor_3d.py
│   # triangulação DLT/SVD
│   # métricas de reprojeção
│
├── tracked_3d.py
│   # Kalman de velocidade constante em XY
│   # associação temporal com Hungarian
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

results/
├── campus_3d.json
├── campus_3d_no_cycle.json
├── shelf_3d.json
├── shelf_3d_no_cycle.json
├── campus_metrics.json
├── campus_metrics_no_cycle.json
├── shelf_metrics.json
├── shelf_metrics_no_cycle.json
└── plots/
    ├── 01_avg_pcp3d.png
    ├── 02_recall.png
    ├── 03_mpjpe.png
    ├── 04_actor_pcp.png
    ├── 05_bone_groups.png
    └── summary.txt
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
| `MATCHER_DISTANCE_D0` | fallback de `matcher_parameters.distance_d0` (datasets usam seu próprio `distance_d0`) | `1.25` |

---