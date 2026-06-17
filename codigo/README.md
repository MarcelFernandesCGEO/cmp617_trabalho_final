# Experimento GeoChat — Tesla V100

Pipeline completo de avaliação semântica de super-resolução de imagens Sentinel-2
por NLP, CLIPScore e VQA com GeoChat-7B.

**Hardware alvo:** 3× Tesla V100-PCIE-32GB (CUDA 12.9)
- GPU 0 e GPU 2: livres para uso
- GPU 1: pode estar em uso por outros jobs — verificar antes

---

## Estrutura esperada

```
experimento_v100/
├── prepare_data.py              # Etapa 1: gera tiles LR/SR/GT
├── geochat_nlp_metrics.py       # Etapa 2: captions + BLEU/ROUGE/TF-IDF
├── geochat_clipscore_metrics.py # Etapa 3: GeoChat + CLIP/RemoteCLIP
├── geochat_vqa_metrics.py       # Etapa 4: VQA estratificada por nível
├── run_pipeline.sh              # Orquestra as 4 etapas
├── geochat/                     # Pacote GeoChat patchado (não alterar)
├── requirements.txt
├── README.md
│
├── data/
│   ├── lr/                      # Imagem Sentinel-2 LR (.jp2 ou .tif)
│   ├── sr/                      # Imagem super-resolvida (.tif)
│   ├── gt/                      # Ortofotos ground truth (.tif, pode ser múltiplos)
│   └── vetores_cobertura_solo.gpkg  # GeoPackage com classes de cobertura do solo
│
└── models/
    └── geochat-7B/              # Transferir do servidor de origem (~14 GB)
```

---

## Setup (uma única vez)

```bash
# 1. Ambiente virtual
python -m venv .venv
source .venv/bin/activate

# 2. Dependências Python
pip install -r requirements.txt

# 3. Pacote GeoChat (com patches para transformers>=4.38)
pip install -e geochat/ --no-deps

# 4. Transferir pesos do modelo
rsync -avz origem:caminho/models/geochat-7B/ models/geochat-7B/
```

---

## Executar

### Pipeline completo (recomendado)

```bash
source .venv/bin/activate

# FP16 nativo — sem quantização, melhor qualidade (V100 32 GB tem espaço de sobra)
CUDA_VISIBLE_DEVICES=0 bash run_pipeline.sh

# Se GPU 0 estiver ocupada, usar GPU 2:
CUDA_VISIBLE_DEVICES=2 bash run_pipeline.sh

# Teste rápido com 5 tiles:
CUDA_VISIBLE_DEVICES=0 bash run_pipeline.sh --max-tiles 5

# Se os tiles já existem (pular prepare_data.py):
CUDA_VISIBLE_DEVICES=0 bash run_pipeline.sh --skip-prepare
```

### Etapas individuais

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0   # ou 2

# Etapa 1 — tiling
python prepare_data.py

# Etapa 2 — NLP (BLEU/ROUGE/TF-IDF)
python geochat_nlp_metrics.py --model-path models/geochat-7B

# Etapa 3 — CLIPScore
python geochat_clipscore_metrics.py --model-path models/geochat-7B

# Etapa 4 — VQA
python geochat_vqa_metrics.py --model-path models/geochat-7B
python geochat_vqa_metrics.py --model-path models/geochat-7B --resume  # retomar se interrompido
```

---

## VRAM necessária

| Modo | VRAM | Recomendação |
|------|------|--------------|
| FP16 (sem flag) | ~14 GB | **Usar na V100 32 GB** — melhor qualidade e velocidade |
| `--load-4bit` | ~5 GB | Só se outra GPU com 32 GB não estiver disponível |
| `--load-8bit` | ~8 GB | Intermediário, raramente necessário |

Com 32 GB de VRAM, o modelo carrega em FP16 com ~18 GB livres para buffers de ativação.
**Não use `--load-4bit` nesta máquina.**

---

## GeoPackage vetorial

O arquivo `data/vetores_cobertura_solo.gpkg` deve ter uma camada por classe de cobertura
do solo. O mapeamento de camadas para nomes de classe está em `LAYER_NAME_MAP` no topo
de cada script. Camadas esperadas:

| Camada no GPKG                    | Classe usada nos scripts |
|-----------------------------------|--------------------------|
| `HID_Massa_Dagua_A`               | massa dagua              |
| `LML_Area_Densamente_Edificada_A` | area edificada           |
| `REL_Terreno_Exposto_A`           | terreno exposto          |
| `VEG_Brejo_Pantano_A`             | brejo pantano            |
| `VEG_Campo_A.shp`                 | campo                    |
| `Veg_Cultivada_A.shp`             | vegetacao cultivada      |
| `VEG_Floresta_A.shp`              | floresta                 |

Se o GeoPackage de uma nova área tiver nomes de camadas diferentes, edite `LAYER_NAME_MAP`
nos scripts.

---

## Perguntas VQA (7 perguntas, 3 níveis)

| ID | Nível | Tipo | Descrição |
|----|-------|------|-----------|
| q01_pct_landcover | 1 | Semântico | % de cada classe: floresta, agricultura, urbano, água, solo exposto |
| q02_has_water | 1 | Semântico | Há corpo d'água? (yes/no) |
| q03_field_count | 2 | Estrutural | Quantos campos agrícolas distintos? |
| q04_field_rows | 2 | Estrutural | Fileiras de cultivo visíveis? (yes/no) |
| q05_building_count | 3 | Detalhe fino | Quantos edifícios visíveis? |
| q06_road_describe | 3 | Detalhe fino | Quantas estradas? Pavimentadas? |
| q07_building_describe | 3 | Detalhe fino | Conte edifícios e descreva telhados |

**Hipótese:** para perguntas de nível 3, SR deve concordar mais com GT do que LR.

---

## Saídas

```
results/
├── tiles/                           # tiles PNG (prepare_data.py)
├── tiles_geochat/                   # composites LR|SR|GT com captions NLP
├── tiles_geochat_clip/              # composites LR|SR|GT com CLIPScores
├── tiles_geochat_vqa/               # composites LR|SR|GT com Q&A
├── geochat_vqa_cache.json           # cache VQA (permite --resume)
├── geochat_metrics_vetorial.csv     # NLP vs GT vetorial
├── geochat_metrics_raster_gt.csv    # NLP vs GT raster
├── geochat_metrics_raster_gt_resampled.csv
├── geochat_clip_vetorial_txt.csv    # CLIPScore text-vs-text vetorial
├── geochat_clip_vetorial_hybrid.csv # CLIPScore hybrid vetorial
├── geochat_clip_raster_gt_txt.csv
├── geochat_clip_raster_gt_hybrid.csv
├── geochat_vqa_answers.csv          # respostas brutas VQA
├── geochat_vqa_metrics.csv          # métricas VQA por tile × pergunta
└── figures/
    ├── geochat_boxplot_*.png
    ├── geochat_scatter_*.png
    ├── gc_clip_*.png
    ├── vqa_sim_by_level.png
    ├── vqa_winrate_heatmap.png
    ├── vqa_delta_by_question.png
    ├── vqa_level_summary.png
    └── vqa_pct_landcover.png
```

---

## Notas importantes

- O cache `geochat_vqa_cache.json` salva cada resposta imediatamente.
  Se o job for interrompido, use `--resume` para continuar de onde parou.
- Use `CUDA_VISIBLE_DEVICES` para selecionar a GPU antes de rodar.
  Verifique o uso atual com `nvidia-smi` — GPU 1 pode estar ocupada.
- Tiles com >50% pixels NoData são descartados pelo `prepare_data.py`.
- GT com 4 bandas (RGBA) é tratado automaticamente — apenas RGB é usado.
- Se o GPKG de uma nova área não tiver cobertura dos tiles, o cenário vetorial
  ficará vazio mas os cenários raster (NLP e CLIPScore) ainda funcionam.
