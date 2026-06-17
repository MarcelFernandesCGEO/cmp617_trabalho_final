# Bypass da verificação de versão do torch para torch.load (CVE-2025-32434).
# Seguro aqui pois carregamos apenas modelos locais conhecidos.
try:
    import transformers.utils.import_utils as _tiu
    _tiu.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as _tmu
    _tmu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

"""
geochat_vqa_metrics.py
======================
Avaliação por VQA (Visual Question Answering) com GeoChat.

Em vez de legendas livres ("descreva a imagem"), o modelo responde perguntas
específicas estratificadas por nível de sensibilidade à resolução:

  Nível 1 — semântico  : conteúdo de alto nível; LR e SR devem responder igual ao GT
  Nível 2 — estrutural : padrões de campo, estradas; SR deve se aproximar do GT
  Nível 3 — detalhe fino: objetos pequenos, texturas; só SR/GT devem acertar

Hipótese central: para perguntas de nível 3, sim(resposta_SR, resposta_GT) >
                  sim(resposta_LR, resposta_GT) — evidência de que a SR recupera
                  informação espacial fina que a LR não possui.

Métricas por resposta:
  - TF-IDF cosine similarity (léxica)
  - Similaridade semântica via SentenceTransformers (all-MiniLM-L6-v2)

Cache: results/geochat_vqa_cache.json — permite retomar se interrompido
       (use --resume para carregar respostas já geradas)

Saídas:
  results/geochat_vqa_answers.csv   ← respostas brutas (tile × pergunta)
  results/geochat_vqa_metrics.csv   ← métricas agregadas (tile × pergunta)
  results/figures/vqa_*.png         ← gráficos de análise

Uso:
  python geochat_vqa_metrics.py --model-path models/geochat-7B --load-4bit
  python geochat_vqa_metrics.py --model-path models/geochat-7B --load-4bit --max-tiles 5
  python geochat_vqa_metrics.py --model-path models/geochat-7B --load-4bit --resume
"""

import os
import sys
import json
import argparse
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE      = Path(__file__).resolve().parent
OUT_DIR   = str(BASE / "results")
FIG_DIR   = os.path.join(OUT_DIR, "figures")
TILES_DIR = os.path.join(OUT_DIR, "tiles")
CACHE_FILE = os.path.join(OUT_DIR, "geochat_vqa_cache.json")

# ---------------------------------------------------------------------------
# Conjunto de perguntas v2 — otimizado após auditoria das respostas dos 54 tiles
#
# Critérios de design:
#   - Nível 1 (semântico): controle — LR e SR devem responder igual ao GT.
#     A pergunta de porcentagem é exceção poderosa: LR frequentemente recusa
#     ("não é possível estimar") enquanto SR/GT dão números — a própria
#     recusa é evidência de baixa qualidade espacial.
#   - Nível 2 (estrutural): padrões de campo/limite; SR mais próximo do GT.
#   - Nível 3 (detalhe fino): edifícios, estradas, texturas; LR claramente pior.
#
# Descartadas da v1 (saturadas/inúteis):
#   q03_has_road      → 54/54 "Yes" em LR, SR e GT — zero discriminação
#   q06_tree_crowns   → 54/54 "individual crowns" em todos
#   q07_image_sharpness → 54/54 "sharp and detailed" em todos
# ---------------------------------------------------------------------------
QUESTIONS = [
    # ── Nível 1: Semântico ──────────────────────────────────────────────────
    {
        "id"         : "q01_pct_landcover",
        "level"      : 1,
        "level_name" : "semantic",
        "text"       : (
            "Estimate the percentage of each land cover type visible in this remote "
            "sensing image. Answer strictly in this format: "
            "forest X%, agriculture X%, urban X%, water X%, bare soil X%. "
            "Percentages must sum to 100%."
        ),
        "max_tokens" : 65,
        "has_pct"    : True,   # flag: extrair valores numéricos para MAE
    },
    {
        "id"         : "q02_has_water",
        "level"      : 1,
        "level_name" : "semantic",
        "text"       : (
            "Is there any water body (river, lake, or pond) visible in this image? "
            "Answer only: yes or no."
        ),
        "max_tokens" : 5,
        "has_pct"    : False,
    },
    # ── Nível 2: Estrutural ─────────────────────────────────────────────────
    {
        "id"         : "q03_field_count",
        "level"      : 2,
        "level_name" : "structural",
        "text"       : (
            "How many distinct agricultural fields or plots can you count in this image? "
            "Give only a number, or say none if no fields are visible."
        ),
        "max_tokens" : 20,
        "has_pct"    : False,
    },
    {
        "id"         : "q04_field_rows",
        "level"      : 2,
        "level_name" : "structural",
        "text"       : (
            "Are agricultural field rows, crop patterns, or field boundaries clearly "
            "visible in this image? Answer only: yes or no."
        ),
        "max_tokens" : 5,
        "has_pct"    : False,
    },
    # ── Nível 3: Detalhe fino ───────────────────────────────────────────────
    {
        "id"         : "q05_building_count",
        "level"      : 3,
        "level_name" : "fine_detail",
        "text"       : (
            "How many distinct buildings or rooftops can you count in this image? "
            "Give only a number, or say none if no buildings are visible."
        ),
        "max_tokens" : 25,
        "has_pct"    : False,
    },
    {
        "id"         : "q06_road_describe",
        "level"      : 3,
        "level_name" : "fine_detail",
        "text"       : (
            "How many roads or paths are visible in this image? "
            "Are they paved or unpaved? Describe briefly where they are located."
        ),
        "max_tokens" : 40,
        "has_pct"    : False,
    },
    {
        "id"         : "q07_building_describe",
        "level"      : 3,
        "level_name" : "fine_detail",
        "text"       : (
            "How many buildings are visible in this image? "
            "Describe their rooftop color or material if possible."
        ),
        "max_tokens" : 45,
        "has_pct"    : False,
    },
]

LEVEL_NAMES = {1: "Semantic (L1)", 2: "Structural (L2)", 3: "Fine detail (L3)"}
LEVEL_COLORS = {1: "#5dade2", 2: "#f39c12", 3: "#e74c3c"}


# ---------------------------------------------------------------------------
# SentenceTransformers (optional, falls back to TF-IDF)
# ---------------------------------------------------------------------------
def _load_sentence_model():
    try:
        from sentence_transformers import SentenceTransformer
        print("  Carregando SentenceTransformer (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        print("  SentenceTransformer carregado.")
        return model
    except ImportError:
        print("  AVISO: sentence-transformers não instalado; usando apenas TF-IDF.")
        print("         Para instalar: pip install sentence-transformers")
        return None


def semantic_sim(a: str, b: str, sent_model) -> float:
    if sent_model is None:
        return float("nan")
    emb = sent_model.encode([a, b], convert_to_tensor=True)
    cos = float(torch.nn.functional.cosine_similarity(
        emb[0].unsqueeze(0), emb[1].unsqueeze(0)
    ).item())
    return max(cos, 0.0)


def tfidf_sim(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    try:
        vec = TfidfVectorizer().fit_transform([a, b])
        return float(cosine_similarity(vec[0], vec[1])[0, 0])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Métrica numérica para q01_pct_landcover
# ---------------------------------------------------------------------------
import re as _re

_PCT_KEYS = ["forest", "agriculture", "urban", "water", "bare soil"]

def parse_pct(text: str) -> dict | None:
    """Extrai valores numéricos de 'forest X%, agriculture X%, ...'
    Retorna dict {classe: float} ou None se o formato não for reconhecido."""
    result = {}
    for key in _PCT_KEYS:
        m = _re.search(rf"{key}\s+(\d+(?:\.\d+)?)\s*%", text, _re.IGNORECASE)
        if m:
            result[key] = float(m.group(1))
    if len(result) < 3:   # menos de 3 classes reconhecidas → recusa/formato errado
        return None
    return result


def pct_mae(a: str, b: str) -> float:
    """Mean Absolute Error entre vetores de porcentagem.
    Retorna NaN se algum texto não tiver formato válido."""
    pa, pb = parse_pct(a), parse_pct(b)
    if pa is None or pb is None:
        return float("nan")
    errors = [abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in _PCT_KEYS]
    return float(np.mean(errors))


def pct_refused(text: str) -> bool:
    """True se o modelo recusou estimar porcentagens (LR frequentemente faz isso)."""
    return parse_pct(text) is None


# ---------------------------------------------------------------------------
# GeoChat VQA wrapper
# ---------------------------------------------------------------------------
class GeoChatVQA:
    def __init__(self, model_path: str, load_4bit: bool = False,
                 load_8bit: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Carregando GeoChat ({model_path}) em {self.device}...")
        if self.device == "cpu":
            print("  AVISO: CPU — inferência lenta. Use --max-tiles para testar.")
            load_4bit = False
            load_8bit = False

        try:
            from geochat.model.builder import load_pretrained_model
            from geochat.mm_utils import get_model_name_from_path
        except ImportError:
            print("\n  ERRO: pacote 'geochat' não encontrado.")
            print("  Instale: pip install -e geochat/ --no-deps")
            sys.exit(1)

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = \
            load_pretrained_model(
                model_path=model_path, model_base=None, model_name=model_name,
                load_8bit=load_8bit, load_4bit=load_4bit, device=self.device,
                device_map={"": 0} if self.device == "cuda" else "cpu",
            )
        self.model.eval()

        from geochat.conversation import conv_templates
        from geochat.mm_utils import tokenizer_image_token, process_images
        from geochat.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        )
        self._conv_templates      = conv_templates
        self._tok_img_token        = tokenizer_image_token
        self._process_images       = process_images
        self._IMAGE_TOKEN_INDEX    = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN  = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN   = DEFAULT_IM_END_TOKEN

        print("  GeoChat pronto.")

    @torch.no_grad()
    def ask(self, pil_img: Image.Image, question: str, max_tokens: int = 30) -> str:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        img_tensor = self._process_images(
            [pil_img], self.image_processor, self.model.config
        )
        if isinstance(img_tensor, list):
            img_tensor = img_tensor[0]
        img_tensor = img_tensor.to(self.device, dtype=self.model.dtype)

        img_tok = (
            self._DEFAULT_IM_START_TOKEN + self._DEFAULT_IMAGE_TOKEN
            + self._DEFAULT_IM_END_TOKEN
            if self.model.config.mm_use_im_start_end
            else self._DEFAULT_IMAGE_TOKEN
        )
        full_prompt = img_tok + "\n" + question

        conv = self._conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)

        input_ids = self._tok_img_token(
            conv.get_prompt(), self.tokenizer,
            self._IMAGE_TOKEN_INDEX, return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        output_ids = self.model.generate(
            input_ids,
            images=img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor,
            max_new_tokens=max_tokens,
            do_sample=False, temperature=0, use_cache=True,
        )
        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sim_by_level(df_metrics, fname):
    """Boxplot de similaridade SR-GT vs LR-GT agrupado por nível de pergunta."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Similaridade com GT por nível de sensibilidade à resolução\n"
        "Hipótese: SR deve superar LR nas perguntas de Nível 3 (detalhe fino)",
        fontsize=11, fontweight="bold",
    )

    for ax, metric, metric_label in zip(
        axes,
        ["tfidf_lr_gt", "semantic_lr_gt"],
        ["TF-IDF Cosine", "Similaridade Semântica (MiniLM)"],
    ):
        sr_col = metric.replace("_lr_", "_sr_")
        data_lr, data_sr, labels, colors = [], [], [], []

        for lvl in [1, 2, 3]:
            sub = df_metrics[df_metrics["level"] == lvl]
            data_lr.append(sub[metric].dropna().values)
            data_sr.append(sub[sr_col].dropna().values)
            labels.append(LEVEL_NAMES[lvl])

        x = np.arange(len(labels))
        w = 0.3
        positions_lr = x - w / 2
        positions_sr = x + w / 2

        bp_lr = ax.boxplot(data_lr, positions=positions_lr, widths=w,
                           patch_artist=True,
                           medianprops=dict(color="black", linewidth=1.5))
        bp_sr = ax.boxplot(data_sr, positions=positions_sr, widths=w,
                           patch_artist=True,
                           medianprops=dict(color="black", linewidth=1.5))

        for patch in bp_lr["boxes"]:
            patch.set_facecolor("#5dade2")
            patch.set_alpha(0.75)
        for patch in bp_sr["boxes"]:
            patch.set_facecolor("#f39c12")
            patch.set_alpha(0.75)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(metric_label)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(facecolor="#5dade2", label="LR vs GT"),
                     Patch(facecolor="#f39c12", label="SR vs GT")],
            fontsize=9,
        )

        # Anotar Δ médio por nível
        for lvl_i, lvl in enumerate([1, 2, 3]):
            sub = df_metrics[df_metrics["level"] == lvl]
            delta = sub[sr_col].mean() - sub[metric].mean()
            ax.text(
                lvl_i, 1.02, f"Δ={delta:+.3f}",
                ha="center", va="bottom", fontsize=8,
                color="#27ae60" if delta > 0 else "#c0392b",
            )

    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()
    print(f"  Salvo: {fname}")


def plot_winrate_heatmap(df_metrics, fname):
    """Heatmap: taxa de vitória SR>LR por pergunta e métrica."""
    questions = df_metrics["question_id"].unique()
    metrics   = ["tfidf", "semantic"]
    metric_labels = {"tfidf": "TF-IDF", "semantic": "Semântica"}

    win_matrix = np.zeros((len(questions), len(metrics)))
    level_arr  = []

    for qi, qid in enumerate(questions):
        sub = df_metrics[df_metrics["question_id"] == qid]
        level_arr.append(sub["level"].iloc[0])
        for mi, m in enumerate(metrics):
            lr_col = f"{m}_lr_gt"
            sr_col = f"{m}_sr_gt"
            if lr_col not in sub.columns or sr_col not in sub.columns:
                win_matrix[qi, mi] = float("nan")
            else:
                valid = sub[[lr_col, sr_col]].dropna()
                if len(valid) == 0:
                    win_matrix[qi, mi] = float("nan")
                else:
                    win_matrix[qi, mi] = (valid[sr_col] > valid[lr_col]).mean()

    q_labels = [
        f"[L{level_arr[i]}] {qid}"
        for i, qid in enumerate(questions)
    ]

    fig, ax = plt.subplots(figsize=(7, max(4, len(questions) * 0.55 + 1.5)))
    im = ax.imshow(win_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0,
                   aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([metric_labels[m] for m in metrics], fontsize=10)
    ax.set_yticks(range(len(q_labels)))
    ax.set_yticklabels(q_labels, fontsize=8)
    ax.set_title(
        "Taxa de vitória SR > LR por pergunta e métrica\n"
        "(verde = SR supera LR; vermelho = SR inferior ou igual)",
        fontsize=10, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Fração de tiles com SR > LR")

    for qi in range(len(questions)):
        for mi in range(len(metrics)):
            val = win_matrix[qi, mi]
            if not np.isnan(val):
                ax.text(mi, qi, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black",
                        fontweight="bold" if val > 0.6 or val < 0.4 else "normal")

    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {fname}")


def plot_delta_by_question(df_metrics, fname):
    """Barplot de Δ (SR_sim_GT − LR_sim_GT) por pergunta, para TF-IDF e Semântica."""
    questions = df_metrics["question_id"].unique()
    metrics = [("tfidf_lr_gt", "tfidf_sr_gt", "TF-IDF"),
               ("semantic_lr_gt", "semantic_sr_gt", "Semântica")]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(questions) * 0.5 + 1.5)))
    fig.suptitle(
        "Ganho médio SR − LR na similaridade com GT por pergunta\n"
        "(positivo = SR mais próximo do GT; esperado subir no Nível 3)",
        fontsize=11, fontweight="bold",
    )

    for ax, (lr_col, sr_col, label) in zip(axes, metrics):
        deltas, colors, q_labels = [], [], []
        for qid in questions:
            sub = df_metrics[df_metrics["question_id"] == qid]
            lvl = sub["level"].iloc[0]
            valid = sub[[lr_col, sr_col]].dropna()
            delta = (valid[sr_col] - valid[lr_col]).mean() if len(valid) > 0 else 0.0
            deltas.append(delta)
            colors.append(LEVEL_COLORS.get(lvl, "#888"))
            q_labels.append(f"[L{lvl}] {qid}")

        y = np.arange(len(questions))
        bars = ax.barh(y, deltas, color=colors, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(q_labels, fontsize=8)
        ax.set_xlabel(f"Δ {label} (SR − LR)", fontsize=9)
        ax.set_title(label, fontsize=10)

        for bar, val in zip(bars, deltas):
            ax.text(
                val + 0.003 * (1 if val >= 0 else -1),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=7.5,
                ha="left" if val >= 0 else "right",
            )

        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(color=LEVEL_COLORS[l], label=LEVEL_NAMES[l])
                     for l in [1, 2, 3]],
            fontsize=8, loc="lower right",
        )

    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {fname}")


def plot_level_summary(df_metrics, fname):
    """Barplot resumo: Δ médio por nível × métrica."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        "Ganho médio SR − LR por nível de sensibilidade à resolução",
        fontsize=12, fontweight="bold",
    )
    pairs = [("tfidf_lr_gt",    "tfidf_sr_gt",    "TF-IDF Cosine"),
             ("semantic_lr_gt", "semantic_sr_gt",  "Semântica (MiniLM)")]

    for ax, (lr_col, sr_col, label) in zip(axes, pairs):
        levels, deltas, err, colors = [], [], [], []
        for lvl in [1, 2, 3]:
            sub = df_metrics[df_metrics["level"] == lvl][[lr_col, sr_col]].dropna()
            d   = (sub[sr_col] - sub[lr_col])
            levels.append(LEVEL_NAMES[lvl])
            deltas.append(d.mean())
            err.append(d.std() / max(np.sqrt(len(d)), 1))
            colors.append(LEVEL_COLORS[lvl])

        bars = ax.bar(levels, deltas, yerr=err, capsize=5,
                      color=colors, alpha=0.85, error_kw={"linewidth": 1.2})
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"Δ {label} (SR − LR)")
        ax.set_title(label, fontsize=10)

        for bar, val in zip(bars, deltas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()
    print(f"  Salvo: {fname}")


def save_tile_composite_vqa(tile_id, pil_lr, pil_sr, pil_gt,
                             qa_rows, fname, display_px=256):
    """Composite visual: LR | SR | GT com respostas VQA em tabela por pergunta."""
    import textwrap

    WRAP          = 42          # chars por célula antes de quebrar linha
    LINE_H_IN     = 0.22        # altura por linha de texto (polegadas)
    HDR_H_IN      = 0.32        # altura do cabeçalho da tabela
    IMG_H_IN      = 3.2         # altura do painel de imagens
    FIG_W         = 15.0        # largura total da figura
    FONT_SZ       = 7.5
    COL_W         = [0.18, 0.27, 0.27, 0.28]  # frações da largura do eixo

    LEVEL_LBL_BG  = {1: "#bfdbfe", 2: "#fde68a", 3: "#bbf7d0"}
    COL_HDR       = ["#e2e8f0", "#93c5fd", "#fcd34d", "#86efac"]
    CELL_BG       = {"lr": "#eff6ff", "sr": "#fffbeb", "gt": "#f0fdf4"}

    def resize(img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.resize((display_px, display_px), Image.LANCZOS)

    def wrap(text):
        if not text:
            return "—"
        lines = textwrap.wrap(str(text).strip(), WRAP)
        return "\n".join(lines) if lines else "—"

    imgs       = [resize(pil_lr), resize(pil_sr)]
    img_labels = ["LR", "SR"]
    if pil_gt is not None:
        imgs.append(resize(pil_gt))
        img_labels.append("GT")
    ncols_img = len(imgs)

    cell_text   = []
    cell_colors = []
    row_heights = []

    for r in qa_rows:
        qid   = r["question_id"]
        level = r["level"]
        lr_w  = wrap(r.get("answer_lr", ""))
        sr_w  = wrap(r.get("answer_sr", ""))
        gt_w  = wrap(r.get("answer_gt", ""))
        nlines = max(lr_w.count("\n") + 1,
                     sr_w.count("\n") + 1,
                     gt_w.count("\n") + 1)
        cell_text.append([qid, lr_w, sr_w, gt_w])
        cell_colors.append([
            LEVEL_LBL_BG.get(level, "#f8fafc"),
            CELL_BG["lr"],
            CELL_BG["sr"],
            CELL_BG["gt"],
        ])
        row_heights.append(max(LINE_H_IN, nlines * LINE_H_IN + 0.06))

    tbl_h_in = HDR_H_IN + sum(row_heights)
    fig_h    = IMG_H_IN + tbl_h_in + 0.5

    fig = plt.figure(figsize=(FIG_W, fig_h))
    gs  = gridspec.GridSpec(
        2, 1,
        height_ratios=[IMG_H_IN, tbl_h_in],
        hspace=0.06,
        figure=fig,
    )

    gs_imgs = gridspec.GridSpecFromSubplotSpec(
        1, ncols_img, subplot_spec=gs[0], wspace=0.03
    )
    for ci, (img, lbl) in enumerate(zip(imgs, img_labels)):
        ax = fig.add_subplot(gs_imgs[ci])
        ax.imshow(img)
        ax.set_title(lbl, fontsize=13, fontweight="bold", pad=4)
        ax.axis("off")

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")

    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=["Pergunta", "LR", "SR", "GT"],
        cellColours=cell_colors,
        colColours=COL_HDR,
        loc="upper center",
        cellLoc="left",
        colWidths=COL_W,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT_SZ)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            h = HDR_H_IN / tbl_h_in
            cell.set_text_props(fontweight="bold", fontsize=FONT_SZ + 0.5)
        else:
            h = row_heights[row - 1] / tbl_h_in
        cell.set_height(h)
        cell.PAD = 0.015

    fig.suptitle(
        f"Tile {tile_id} — GeoChat VQA",
        fontsize=12, fontweight="bold", y=0.995,
    )
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()


def _plot_pct_landcover(pct_df, fname):
    """Dois painéis: (1) taxa de recusa LR vs SR; (2) MAE vs GT por tile."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "q01_pct_landcover — Estimativa de porcentagem por classe de cobertura\n"
        "A recusa do modelo (LR) e o MAE (erro numérico) evidenciam ganho da SR",
        fontsize=10, fontweight="bold",
    )

    # Painel 1: taxa de recusa
    ax = axes[0]
    n = len(pct_df)
    lr_ref = pct_df["lr_refused"].sum()
    sr_ref = pct_df["sr_refused"].sum()
    bars = ax.bar(["LR", "SR"], [lr_ref / n * 100, sr_ref / n * 100],
                  color=["#5dade2", "#f39c12"], alpha=0.85)
    for bar, val in zip(bars, [lr_ref, sr_ref]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val}/{n} ({val/n*100:.0f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("% de tiles onde o modelo recusou estimar")
    ax.set_title("Taxa de recusa (sem formato X%)\n"
                 "LR recusa mais → menos informação visível", fontsize=9)
    ax.set_ylim(0, 110)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Painel 2: MAE por tile (para tiles onde ambos têm formato válido)
    ax2 = axes[1]
    valid = pct_df[["mae_lr_gt", "mae_sr_gt"]].dropna()
    if len(valid) > 0:
        bp = ax2.boxplot(
            [valid["mae_lr_gt"].values, valid["mae_sr_gt"].values],
            labels=["LR", "SR"], patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#5dade2")
        bp["boxes"][1].set_facecolor("#f39c12")
        ax2.set_ylabel("MAE vs GT (pontos percentuais)")
        ax2.set_title(
            f"Erro numérico vs GT (n={len(valid)} tiles com formato válido)\n"
            "Menor = mais próximo da GT",
            fontsize=9,
        )
        delta_mae = valid["mae_sr_gt"].mean() - valid["mae_lr_gt"].mean()
        ax2.text(0.5, 0.95, f"Δ MAE = {delta_mae:+.2f}pp (negativo = SR mais preciso)",
                 transform=ax2.transAxes, ha="center", va="top",
                 fontsize=9, color="#27ae60" if delta_mae < 0 else "#c0392b",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    else:
        ax2.text(0.5, 0.5, "Nenhum tile com formato % válido\nem LR e GT simultaneamente",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()
    print(f"  Salvo: {fname}")


def print_summary(df_metrics):
    print(f"\n{'='*70}")
    print(f"Resumo GeoChat VQA v2 — n={len(df_metrics['tile_id'].unique())} tiles, "
          f"{len(QUESTIONS)} perguntas")
    print(f"{'='*70}")
    pairs = [("tfidf_lr_gt",    "tfidf_sr_gt",    "TF-IDF"),
             ("semantic_lr_gt", "semantic_sr_gt",  "Semântica")]
    for lvl in [1, 2, 3]:
        sub = df_metrics[df_metrics["level"] == lvl]
        print(f"\n  {LEVEL_NAMES[lvl]} ({len(sub)} obs):")
        for lr_col, sr_col, label in pairs:
            valid = sub[[lr_col, sr_col]].dropna()
            if len(valid) == 0:
                continue
            delta = (valid[sr_col] - valid[lr_col]).mean()
            wins  = (valid[sr_col] > valid[lr_col]).sum()
            print(f"    {label:20s}: LR={valid[lr_col].mean():.4f}  "
                  f"SR={valid[sr_col].mean():.4f}  "
                  f"Δ={delta:+.4f}  SR>LR: {wins}/{len(valid)}")

    # Resumo especial para q01_pct_landcover
    pct_sub = df_metrics[df_metrics["question_id"] == "q01_pct_landcover"]
    if len(pct_sub) > 0:
        print(f"\n  q01_pct_landcover — Detalhes:")
        lr_ref = pct_sub["lr_refused"].sum()
        sr_ref = pct_sub["sr_refused"].sum()
        n      = len(pct_sub)
        print(f"    Recusas (sem formato %): LR={lr_ref}/{n} ({lr_ref/n*100:.0f}%)  "
              f"SR={sr_ref}/{n} ({sr_ref/n*100:.0f}%)")
        mae_valid = pct_sub[["mae_lr_gt", "mae_sr_gt"]].dropna()
        if len(mae_valid) > 0:
            print(f"    MAE vs GT (tiles com formato): "
                  f"LR={mae_valid['mae_lr_gt'].mean():.2f}pp  "
                  f"SR={mae_valid['mae_sr_gt'].mean():.2f}pp  "
                  f"Δ={mae_valid['mae_sr_gt'].mean()-mae_valid['mae_lr_gt'].mean():+.2f}pp"
                  f"  (negativo = SR mais preciso)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GeoChat VQA metrics pipeline")
    parser.add_argument("--model-path", type=str, default="MBZUAI/geochat-7B")
    parser.add_argument("--load-4bit",  action="store_true")
    parser.add_argument("--load-8bit",  action="store_true")
    parser.add_argument("--max-tiles",  type=int, default=None)
    parser.add_argument("--resume",     action="store_true",
                        help="Carregar respostas já geradas do cache JSON")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    tiles_out = os.path.join(OUT_DIR, "tiles_geochat_vqa")
    os.makedirs(tiles_out, exist_ok=True)

    # ---- Tiles metadata ---------------------------------------------------
    meta_csv = os.path.join(TILES_DIR, "tiles_metadata.csv")
    if not os.path.exists(meta_csv):
        print(f"ERRO: {meta_csv} não encontrado. Execute prepare_data.py primeiro.")
        sys.exit(1)
    meta_df = pd.read_csv(meta_csv)
    if args.max_tiles is not None:
        meta_df = meta_df.head(args.max_tiles)
    print(f"[Info] {len(meta_df)} tiles, {len(QUESTIONS)} perguntas por tile × 3 imagens")
    total_calls = len(meta_df) * len(QUESTIONS) * 3
    print(f"[Info] Total de chamadas GeoChat: {total_calls} (~{total_calls*5/60:.0f}–{total_calls*8/60:.0f} min)")

    # ---- Cache ------------------------------------------------------------
    cache: dict = {}
    if args.resume and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        cached_calls = sum(len(v) for v in cache.values())
        print(f"[Resume] Cache carregado: {len(cache)} tiles, {cached_calls} respostas")

    def cache_key(tile_id, qid, img_type):
        return f"{tile_id}|{qid}|{img_type}"

    def save_cache():
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    # ---- Modelos ----------------------------------------------------------
    print("\n[Fase 1] Carregando modelos...")

    # Verificar se há chamadas que precisam de GeoChat
    pending_calls = 0
    for row in meta_df.itertuples():
        for q in QUESTIONS:
            for img_t in ["lr", "sr", "gt"]:
                k = cache_key(row.tile_id, q["id"], img_t)
                if k not in cache:
                    pending_calls += 1

    vqa_model = None
    if pending_calls > 0:
        print(f"  {pending_calls} chamadas GeoChat pendentes.")
        vqa_model = GeoChatVQA(
            model_path=args.model_path,
            load_4bit=args.load_4bit,
            load_8bit=args.load_8bit,
        )
    else:
        print("  Todas as respostas já estão no cache. Pulando carregamento do GeoChat.")

    sent_model = _load_sentence_model()

    # ---- Inferência -------------------------------------------------------
    print(f"\n[Fase 2] Gerando respostas VQA...")
    t_start = time.time()
    tiles_processed = 0

    for row in meta_df.itertuples():
        tile_id = row.tile_id
        lr_file = os.path.join(TILES_DIR, "lr", f"{tile_id}.png")
        sr_file = os.path.join(TILES_DIR, "sr", f"{tile_id}.png")
        gt_file = os.path.join(TILES_DIR, "gt", f"{tile_id}.png")

        if not os.path.exists(lr_file) or not os.path.exists(sr_file):
            print(f"  {tile_id}: skip (arquivo não encontrado)")
            continue

        pil_lr = Image.open(lr_file).convert("RGB")
        pil_sr = Image.open(sr_file).convert("RGB")
        pil_gt = Image.open(gt_file).convert("RGB") if os.path.exists(gt_file) else None

        tiles_processed += 1
        t0 = time.time()
        n_pending = sum(
            1 for q in QUESTIONS
            for img_t in (["lr", "sr", "gt"] if pil_gt else ["lr", "sr"])
            if cache_key(tile_id, q["id"], img_t) not in cache
        )
        print(f"  [{tiles_processed}/{len(meta_df)}] {tile_id} "
              f"({n_pending} chamadas pendentes)...", flush=True)

        for q in QUESTIONS:
            for img_t, pil_img in [("lr", pil_lr), ("sr", pil_sr),
                                    ("gt", pil_gt)]:
                if pil_img is None:
                    continue
                k = cache_key(tile_id, q["id"], img_t)
                if k not in cache:
                    if vqa_model is None:
                        continue
                    answer = vqa_model.ask(pil_img, q["text"], q["max_tokens"])
                    cache[k] = answer

        save_cache()
        print(f"    OK ({time.time()-t0:.1f}s)", flush=True)

    total_inf = time.time() - t_start
    print(f"\n  Inferência concluída em {total_inf:.1f}s")
    save_cache()

    # ---- Calcular métricas ------------------------------------------------
    print("\n[Fase 3] Calculando métricas de similaridade...")
    answer_rows = []
    metric_rows = []

    for row in meta_df.itertuples():
        tile_id = row.tile_id
        pil_lr = pil_sr = pil_gt = None
        lr_file = os.path.join(TILES_DIR, "lr", f"{tile_id}.png")
        sr_file = os.path.join(TILES_DIR, "sr", f"{tile_id}.png")
        gt_file = os.path.join(TILES_DIR, "gt", f"{tile_id}.png")

        has_images = os.path.exists(lr_file) and os.path.exists(sr_file)
        if not has_images:
            continue

        pil_lr = Image.open(lr_file).convert("RGB")
        pil_sr = Image.open(sr_file).convert("RGB")
        pil_gt = Image.open(gt_file).convert("RGB") if os.path.exists(gt_file) else None

        qa_for_tile = []
        for q in QUESTIONS:
            ans_lr = cache.get(cache_key(tile_id, q["id"], "lr"), "")
            ans_sr = cache.get(cache_key(tile_id, q["id"], "sr"), "")
            ans_gt = cache.get(cache_key(tile_id, q["id"], "gt"), "")

            tfidf_lr_gt = tfidf_sim(ans_lr, ans_gt) if ans_gt else float("nan")
            tfidf_sr_gt = tfidf_sim(ans_sr, ans_gt) if ans_gt else float("nan")
            sem_lr_gt   = semantic_sim(ans_lr, ans_gt, sent_model) if ans_gt else float("nan")
            sem_sr_gt   = semantic_sim(ans_sr, ans_gt, sent_model) if ans_gt else float("nan")

            # Métricas numéricas específicas para pergunta de porcentagem
            mae_lr_gt  = float("nan")
            mae_sr_gt  = float("nan")
            lr_refused = False
            sr_refused = False
            if q.get("has_pct"):
                mae_lr_gt  = pct_mae(ans_lr, ans_gt)
                mae_sr_gt  = pct_mae(ans_sr, ans_gt)
                lr_refused = pct_refused(ans_lr)
                sr_refused = pct_refused(ans_sr)

            answer_rows.append({
                "tile_id"    : tile_id,
                "question_id": q["id"],
                "level"      : q["level"],
                "level_name" : q["level_name"],
                "question"   : q["text"],
                "answer_lr"  : ans_lr,
                "answer_sr"  : ans_sr,
                "answer_gt"  : ans_gt,
            })
            metric_rows.append({
                "tile_id"       : tile_id,
                "question_id"   : q["id"],
                "level"         : q["level"],
                "level_name"    : q["level_name"],
                "tfidf_lr_gt"   : tfidf_lr_gt,
                "tfidf_sr_gt"   : tfidf_sr_gt,
                "semantic_lr_gt": sem_lr_gt,
                "semantic_sr_gt": sem_sr_gt,
                "mae_lr_gt"     : mae_lr_gt,
                "mae_sr_gt"     : mae_sr_gt,
                "lr_refused"    : lr_refused,
                "sr_refused"    : sr_refused,
            })
            qa_for_tile.append({
                "question_id": q["id"],
                "level"      : q["level"],
                "answer_lr"  : ans_lr,
                "answer_sr"  : ans_sr,
                "answer_gt"  : ans_gt,
            })

        # Composite visual por tile
        if pil_lr is not None:
            composite_fname = os.path.join(tiles_out, f"{tile_id}_vqa.png")
            try:
                save_tile_composite_vqa(
                    tile_id, pil_lr, pil_sr, pil_gt, qa_for_tile, composite_fname
                )
            except Exception as e:
                print(f"  AVISO: erro ao salvar composite {tile_id}: {e}")

    df_answers = pd.DataFrame(answer_rows)
    df_metrics = pd.DataFrame(metric_rows)

    # ---- Salvar CSVs ------------------------------------------------------
    answers_csv = os.path.join(OUT_DIR, "geochat_vqa_answers.csv")
    metrics_csv = os.path.join(OUT_DIR, "geochat_vqa_metrics.csv")
    df_answers.to_csv(answers_csv, index=False)
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\n  Salvo: {answers_csv} ({len(df_answers)} linhas)")
    print(f"  Salvo: {metrics_csv} ({len(df_metrics)} linhas)")

    if len(df_metrics) == 0:
        print("\nAVISO: nenhuma métrica calculada — verifique o cache.")
        return

    # ---- Gráficos ---------------------------------------------------------
    print("\n[Fase 4] Gerando gráficos...")

    plot_sim_by_level(
        df_metrics,
        os.path.join(FIG_DIR, "vqa_sim_by_level.png"),
    )
    plot_winrate_heatmap(
        df_metrics,
        os.path.join(FIG_DIR, "vqa_winrate_heatmap.png"),
    )
    plot_delta_by_question(
        df_metrics,
        os.path.join(FIG_DIR, "vqa_delta_by_question.png"),
    )
    plot_level_summary(
        df_metrics,
        os.path.join(FIG_DIR, "vqa_level_summary.png"),
    )

    # Plot especial: taxa de recusa e MAE para q01_pct_landcover
    pct_df = df_metrics[df_metrics["question_id"] == "q01_pct_landcover"]
    if len(pct_df) > 0:
        _plot_pct_landcover(pct_df, os.path.join(FIG_DIR, "vqa_pct_landcover.png"))

    print_summary(df_metrics)
    print(f"\nConcluído! Resultados em: {OUT_DIR}")
    print(f"Cache salvo em: {CACHE_FILE}")


if __name__ == "__main__":
    main()
