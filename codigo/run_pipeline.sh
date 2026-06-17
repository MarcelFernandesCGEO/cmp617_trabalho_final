#!/usr/bin/env bash
# run_pipeline.sh — Pipeline completo de avaliação semântica SR (genérico)
#
# Executa uma região por vez. Os caminhos são todos relativos a este script,
# sem dependência de servidor ou caminhos absolutos. Para consolidar duas
# regiões e gerar as figuras do artigo, use depois:
#   python consolidar_regioes.py
#   python gerar_figuras_consolidadas.py
#
# Uso:
#   bash run_pipeline.sh                          # FP16 nativo (GPU com >= ~14 GB)
#   bash run_pipeline.sh --load-4bit              # quantização 4-bit (GPU < ~6 GB)
#   bash run_pipeline.sh --max-tiles 5            # teste rápido com 5 tiles
#   bash run_pipeline.sh --skip-prepare           # pular tiling (tiles já existem)
#   bash run_pipeline.sh --resume                 # pular tiles já processados, concatenar CSVs
#   bash run_pipeline.sh --skip-prepare --resume  # rodar apenas os tiles novos sem re-tiling
#
# Entradas esperadas:
#   data/lr/          ← imagem Sentinel-2 LR (.jp2 ou .tif)
#   data/sr/          ← imagem super-resolvida (.tif)
#   data/gt/          ← ortofotos ground truth (.tif, podem ser múltiplos)
#   data/vetores_cobertura_solo.gpkg  ← GeoPackage com classes de cobertura do solo
#   models/geochat-7B/  ← pesos do GeoChat-7B (~14 GB, NÃO incluídos nesta entrega;
#                          obtenha-os do repositório oficial do GeoChat)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ativar venv se existir e python não estiver no PATH correto
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ] && [ "$(which python 2>/dev/null)" != "$SCRIPT_DIR/.venv/bin/python" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

MODEL_PATH="models/geochat-7B"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

export NLTK_DATA="$SCRIPT_DIR/nltk_data"

# ── Separar argumentos por destino ───────────────────────────────────────────
# BLIP_ARGS    → blip_nlp_metrics.py, blip_clipscore_metrics.py
#                aceita: --max-tiles, --resume
# GEOCHAT_ARGS → geochat_*.py
#                aceita: --max-tiles, --resume, --model-path, --load-4bit, --load-8bit
SKIP_PREPARE=0
BLIP_ARGS=()
GEOCHAT_ARGS=()
_SKIP_NEXT=0

for arg in "$@"; do
    if [ $_SKIP_NEXT -eq 1 ]; then
        GEOCHAT_ARGS+=("$arg")   # valor do argumento anterior (ex: path do modelo)
        _SKIP_NEXT=0
        continue
    fi
    case "$arg" in
        --skip-prepare)
            SKIP_PREPARE=1 ;;
        --max-tiles|--resume)
            BLIP_ARGS+=("$arg")
            GEOCHAT_ARGS+=("$arg") ;;
        --load-4bit|--load-8bit)
            GEOCHAT_ARGS+=("$arg") ;;
        --model-path)
            GEOCHAT_ARGS+=("$arg")
            _SKIP_NEXT=1 ;;
        --max-tiles=*|--resume=*)
            BLIP_ARGS+=("$arg")
            GEOCHAT_ARGS+=("$arg") ;;
        *)
            echo "AVISO: argumento desconhecido ignorado: $arg" ;;
    esac
done

echo "========================================================"
echo " Pipeline de Avaliação Semântica SR — GeoChat"
echo " Diretório    : $SCRIPT_DIR"
echo " Modelo       : $MODEL_PATH"
echo " Args BLIP    : ${BLIP_ARGS[*]:-nenhum}"
echo " Args GeoChat : ${GEOCHAT_ARGS[*]:-nenhum}"
echo " GPU          : ${CUDA_VISIBLE_DEVICES:-auto (todas)}"
echo "========================================================"

# Aviso se CUDA_VISIBLE_DEVICES não foi definido
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo ""
    echo "AVISO: CUDA_VISIBLE_DEVICES não definido."
    echo "  Para fixar uma GPU específica: CUDA_VISIBLE_DEVICES=0 bash run_pipeline.sh"
    echo "  Continuando com seleção automática de GPU..."
    echo ""
fi

# ── Verificar entradas ───────────────────────────────────────────────────────
echo ""
echo "[Check] Verificando dados de entrada..."
LR_COUNT=$(find data/lr  -name "*.jp2" -o -name "*.tif" 2>/dev/null | wc -l)
SR_COUNT=$(find data/sr  -name "*.tif"                  2>/dev/null | wc -l)
GT_COUNT=$(find data/gt  -name "*.tif"                  2>/dev/null | wc -l)
GPKG_OK=0; [ -f "data/vetores_cobertura_solo.gpkg" ] && GPKG_OK=1
MODEL_OK=0; [ -d "$MODEL_PATH" ] && MODEL_OK=1

echo "  LR     : $LR_COUNT arquivo(s) em data/lr/"
echo "  SR     : $SR_COUNT arquivo(s) em data/sr/"
echo "  GT     : $GT_COUNT arquivo(s) em data/gt/"
echo "  GPKG   : $([ $GPKG_OK -eq 1 ] && echo 'OK' || echo 'AUSENTE')"
echo "  Modelo : $([ $MODEL_OK -eq 1 ] && echo 'OK' || echo 'AUSENTE — transfira os pesos')"

if [ "$LR_COUNT" -eq 0 ] || [ "$SR_COUNT" -eq 0 ] || [ "$GT_COUNT" -eq 0 ]; then
    echo ""
    echo "ERRO: Faltam arquivos de imagem em data/lr/, data/sr/ ou data/gt/"
    exit 1
fi
if [ $MODEL_OK -eq 0 ]; then
    echo ""
    echo "ERRO: Modelo não encontrado em $MODEL_PATH"
    echo "  Os pesos do GeoChat-7B (~14 GB) não acompanham a entrega."
    echo "  Baixe-os do repositório oficial do GeoChat e coloque em $MODEL_PATH/"
    exit 1
fi

# ── Etapa 1 — Tiling ────────────────────────────────────────────────────────
if [ $SKIP_PREPARE -eq 0 ]; then
    echo ""
    echo "========================================================"
    echo " Etapa 1 — prepare_data.py (tiling LR/SR/GT)"
    echo "========================================================"
    python prepare_data.py 2>&1 | tee "$LOG_DIR/etapa1_prepare.log"
    TILE_COUNT=$(find results/tiles/lr -name "*.png" 2>/dev/null | wc -l)
    echo "  Tiles gerados: $TILE_COUNT"
    if [ "$TILE_COUNT" -eq 0 ]; then
        echo "ERRO: Nenhum tile gerado — verifique os bounds das imagens."
        exit 1
    fi
else
    echo ""
    echo "[Etapa 1 pulada — --skip-prepare]"
    TILE_COUNT=$(find results/tiles/lr -name "*.png" 2>/dev/null | wc -l)
    echo "  Tiles existentes: $TILE_COUNT"
fi

# ── Etapa 2 — BLIP NLP (Cenários A, B, C) ───────────────────────────────────
echo ""
echo "========================================================"
echo " Etapa 2 — blip_nlp_metrics.py (BLIP + BLEU/ROUGE/TF-IDF)"
echo "========================================================"
python blip_nlp_metrics.py \
    "${BLIP_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/etapa2_blip_nlp.log"

# ── Etapa 3 — BLIP CLIPScore (Cenários A, B, C) ─────────────────────────────
echo ""
echo "========================================================"
echo " Etapa 3 — blip_clipscore_metrics.py (BLIP + CLIP/RemoteCLIP)"
echo "========================================================"
python blip_clipscore_metrics.py \
    "${BLIP_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/etapa3_blip_clipscore.log"

# ── Etapa 4 — GeoChat NLP ───────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Etapa 4 — geochat_nlp_metrics.py (captions + BLEU/ROUGE/TF-IDF)"
echo "========================================================"
python geochat_nlp_metrics.py \
    --model-path "$MODEL_PATH" \
    "${GEOCHAT_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/etapa4_nlp.log"

# ── Etapa 5 — GeoChat CLIPScore ─────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Etapa 5 — geochat_clipscore_metrics.py (GeoChat + CLIP/RemoteCLIP)"
echo "========================================================"
python geochat_clipscore_metrics.py \
    --model-path "$MODEL_PATH" \
    "${GEOCHAT_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/etapa5_clipscore.log"

# ── Etapa 6 — GeoChat VQA ───────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Etapa 6 — geochat_vqa_metrics.py (VQA por nível de resolução)"
echo "========================================================"
python geochat_vqa_metrics.py \
    --model-path "$MODEL_PATH" \
    "${GEOCHAT_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/etapa6_vqa.log"

# ── Etapa 7 — VQA × GeoPackage ──────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Etapa 7 — geochat_vqa_vetorial.py (MAE q01 vs GeoPackage)"
echo "========================================================"
python geochat_vqa_vetorial.py \
    2>&1 | tee "$LOG_DIR/etapa7_vqa_vetorial.log"

# ── Resumo ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " PIPELINE CONCLUÍDO"
echo "========================================================"
echo ""
echo "Resultados em: $SCRIPT_DIR/results/"
echo ""
echo "CSVs gerados:"
find results -name "*.csv" \
    -not -path "*/.venv/*" \
    -not -path "*/tiles/*" \
    | sort | while read -r f; do
    LINES=$(wc -l < "$f")
    echo "  $f  ($((LINES - 1)) registros)"
done
echo ""
echo "Figuras geradas:"
find results/figures -name "*.png" 2>/dev/null | sort | while read -r f; do
    echo "  $f"
done
echo ""
echo "Logs em: $LOG_DIR/"
echo ""
echo "Tiles composites (VQA):"
find results -name "*_vqa.png" 2>/dev/null | wc -l | xargs -I{} echo "  {} tiles com composite VQA gerado"
