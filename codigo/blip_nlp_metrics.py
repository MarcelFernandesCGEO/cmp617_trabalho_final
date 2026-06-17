"""
blip_nlp_metrics.py
===================
Avaliação NLP com BLIP para 575 tiles — Cenários A (GT vetorial) e B (GT raster).

Lê tiles PNG já gerados pelo prepare_data.py e tiles_metadata.csv para bboxes.
Usa o mesmo GeoPackage do pipeline GeoChat para classes vetoriais.

Saídas:
  results/blip_metrics_vetorial.csv      — Cenário A (legenda vs nome da classe)
  results/blip_metrics_raster_gt.csv     — Cenário B (legenda LR/SR vs legenda GT)
  results/blip_metrics_raster_gt_resampled.csv — Cenário C (idem, GT reamostrado)
  results/figures/blip_*.png

Uso:
  python blip_nlp_metrics.py
  python blip_nlp_metrics.py --max-tiles 10
  python blip_nlp_metrics.py --resume
"""

import os
import sys
import argparse
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers.utils import import_utils as _iu
import transformers.modeling_utils as _mu
_iu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None
from transformers import BlipProcessor, BlipForConditionalGeneration
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import geopandas as gpd
import pyogrio

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE      = Path(__file__).resolve().parent
TILES_DIR = BASE / "results" / "tiles"
META_CSV  = TILES_DIR / "tiles_metadata.csv"
GPKG_PATH = BASE / "data" / "vetores_cobertura_solo.gpkg"
OUT_DIR   = BASE / "results"
FIG_DIR   = OUT_DIR / "figures"

os.makedirs(FIG_DIR, exist_ok=True)

# GeoPackage layer → English class name (para BLIP em inglês)
LAYER_CLASS_MAP = {
    "VEG_Floresta_A.shp":               "forest",
    "VEG_Campo_A.shp":                  "field",
    "Veg_Cultivada_A.shp":              "cultivated vegetation",
    "HID_Massa_Dagua_A":                "water body",
    "REL_Terreno_Exposto_A":            "bare ground",
    "VEG_Brejo_Pantano_A":              "bare ground",
    "LML_Area_Densamente_Edificada_A":  "urban area",
}

# ---------------------------------------------------------------------------
# NLTK
# ---------------------------------------------------------------------------
_nltk_dir = os.environ.get("NLTK_DATA")
if _nltk_dir and _nltk_dir not in nltk.data.path:
    nltk.data.path.insert(0, _nltk_dir)
for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True, raise_on_error=False)

# ---------------------------------------------------------------------------
# NLP metrics
# ---------------------------------------------------------------------------
_smoother = SmoothingFunction().method1

def bleu1(hyp: str, ref: str) -> float:
    h, r = hyp.lower().split(), ref.lower().split()
    if not h or not r:
        return 0.0
    return sentence_bleu([r], h, weights=(1, 0, 0, 0), smoothing_function=_smoother)

def rouge1(hyp: str, ref: str) -> float:
    sc = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    return sc.score(ref, hyp)["rouge1"].fmeasure

def tfidf_cosine(hyp: str, ref: str) -> float:
    try:
        v = TfidfVectorizer().fit_transform([hyp, ref])
        return float(cosine_similarity(v[0], v[1])[0, 0])
    except Exception:
        return 0.0

def compute_metrics(hyp: str, ref: str) -> dict:
    return {"bleu1": bleu1(hyp, ref), "rouge1": rouge1(hyp, ref), "tfidf": tfidf_cosine(hyp, ref)}

# ---------------------------------------------------------------------------
# BLIP
# ---------------------------------------------------------------------------
def load_blip(device: str):
    print("  Carregando BLIP (local_files_only)...")
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(
        "Salesforce/blip-image-captioning-base", local_files_only=True
    )
    proc = BlipProcessor.from_pretrained(model_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    return proc, model

@torch.no_grad()
def caption(pil_img: Image.Image, proc, model, device: str) -> str:
    inputs = proc(images=pil_img.convert("RGB"), return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=40)
    return proc.decode(out[0], skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# GeoPackage: classe dominante por bbox
# ---------------------------------------------------------------------------
def load_gpkg(gpkg_path: Path) -> dict:
    """Retorna {layer_name: GeoDataFrame} em EPSG:3857.
    Resolve variantes de nome com/sem sufixo .shp."""
    # Monta lookup tolerante: normaliza chaves do mapa (sem .shp, lowercase)
    norm_map = {k.removesuffix(".shp").lower(): (k, v)
                for k, v in LAYER_CLASS_MAP.items()}
    gdfs = {}
    for name, _ in pyogrio.list_layers(str(gpkg_path)):
        key = name.removesuffix(".shp").lower()
        if key in norm_map:
            gdf = gpd.read_file(str(gpkg_path), layer=name).to_crs("EPSG:3857")
            orig_key, _ = norm_map[key]
            gdfs[orig_key] = gdf  # usa chave canônica do mapa para dominant_class()
    return gdfs

def dominant_class(bbox: tuple, gdfs: dict) -> str | None:
    from shapely.geometry import box as shapely_box
    tile_geom = shapely_box(*bbox)
    tile_area = tile_geom.area
    best_cls, best_area = None, 0.0
    seen_classes = {}
    for layer_name, gdf in gdfs.items():
        cls = LAYER_CLASS_MAP[layer_name]
        try:
            area = gdf.geometry.intersection(tile_geom).area.sum()
        except Exception:
            area = 0.0
        seen_classes[cls] = seen_classes.get(cls, 0.0) + area
    for cls, area in seen_classes.items():
        if area > best_area:
            best_area = area
            best_cls = cls
    if best_area < 0.05 * tile_area:
        return None
    return best_cls

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
METRICS = ["bleu1", "rouge1", "tfidf"]
LABELS  = {"bleu1": "BLEU-1", "rouge1": "ROUGE-1 F1", "tfidf": "TF-IDF Cosine"}

def plot_boxplots(df: pd.DataFrame, scenario: str, fname: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"BLIP NLP — {scenario}", fontsize=13)
    for ax, m in zip(axes, METRICS):
        data = [df[f"{m}_lr"].dropna().values, df[f"{m}_sr"].dropna().values]
        bp = ax.boxplot(data, labels=["LR", "SR"], patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("#5dade2")
        bp["boxes"][1].set_facecolor("#f39c12")
        ax.set_title(LABELS[m])
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()

def plot_barplot_by_class(df: pd.DataFrame, fname: Path):
    classes = df["classe"].dropna().unique()
    if len(classes) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(classes) * 3), 5))
    fig.suptitle("BLIP NLP por Classe — GT Vetorial", fontsize=13)
    for ax, m in zip(axes, METRICS):
        lr_m = [df[df["classe"] == c][f"{m}_lr"].mean() for c in classes]
        sr_m = [df[df["classe"] == c][f"{m}_sr"].mean() for c in classes]
        x = np.arange(len(classes))
        ax.bar(x - 0.175, lr_m, 0.35, label="LR", color="#5dade2")
        ax.bar(x + 0.175, sr_m, 0.35, label="SR", color="#f39c12")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_title(LABELS[m])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()

def print_summary(df: pd.DataFrame, scenario: str):
    n = len(df)
    print(f"\n{'='*60}")
    print(f"Resumo BLIP NLP — {scenario}  (n={n})")
    print(f"{'='*60}")
    for m in METRICS:
        lr = df[f"{m}_lr"].dropna()
        sr = df[f"{m}_sr"].dropna()
        wins = (df[f"{m}_sr"] > df[f"{m}_lr"]).sum()
        print(f"  {LABELS[m]:20s}  LR={lr.mean():.4f}  SR={sr.mean():.4f}  "
              f"Δ={sr.mean()-lr.mean():+.4f}  SR>LR={wins}/{n}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Pula tiles já presentes nos CSVs de saída")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    # ── Verificações ─────────────────────────────────────────────────────────
    for p, name in [(META_CSV, "tiles_metadata.csv"), (GPKG_PATH, "GeoPackage")]:
        if not p.exists():
            print(f"ERRO: {name} não encontrado em {p}")
            sys.exit(1)

    meta_df = pd.read_csv(META_CSV)
    if args.max_tiles:
        meta_df = meta_df.head(args.max_tiles)

    # ── Resume: carregar tiles já processados ─────────────────────────────────
    csv_vec = OUT_DIR / "blip_metrics_vetorial.csv"
    csv_gt  = OUT_DIR / "blip_metrics_raster_gt.csv"
    csv_rs  = OUT_DIR / "blip_metrics_raster_gt_resampled.csv"

    done_ids = set()
    if args.resume:
        for csv in [csv_vec, csv_gt, csv_rs]:
            if csv.exists():
                done_ids.update(pd.read_csv(csv)["tile_id"].tolist())
        print(f"  Resume: {len(done_ids)} tile_ids já processados (de qualquer CSV)")

    # ── Carregar GeoPackage ───────────────────────────────────────────────────
    print("Carregando GeoPackage...")
    gdfs = load_gpkg(GPKG_PATH)
    print(f"  {len(gdfs)} camadas carregadas")

    # ── Carregar BLIP ─────────────────────────────────────────────────────────
    print("Carregando BLIP...")
    proc, model = load_blip(device)

    # ── Processar tiles ───────────────────────────────────────────────────────
    print(f"\nProcessando {len(meta_df)} tiles...")
    records_vec = []
    records_gt  = []
    records_rs  = []

    for i, row in enumerate(meta_df.itertuples(), 1):
        tile_id = row.tile_id

        if args.resume and tile_id in done_ids:
            continue

        lr_path = TILES_DIR / "lr" / f"{tile_id}.png"
        sr_path = TILES_DIR / "sr" / f"{tile_id}.png"
        gt_path = TILES_DIR / "gt" / f"{tile_id}.png"

        if not lr_path.exists() or not sr_path.exists():
            continue

        pil_lr = Image.open(lr_path).convert("RGB")
        pil_sr = Image.open(sr_path).convert("RGB")

        cap_lr = caption(pil_lr, proc, model, device)
        cap_sr = caption(pil_sr, proc, model, device)

        # Cenário A — GT vetorial
        bbox = (row.x0, row.y0, row.x1, row.y1)
        cls = dominant_class(bbox, gdfs)
        if cls is not None:
            m_lr = compute_metrics(cap_lr, cls)
            m_sr = compute_metrics(cap_sr, cls)
            records_vec.append({
                "tile_id": tile_id, "classe": cls,
                "caption_lr": cap_lr, "caption_sr": cap_sr,
                "bleu1_lr": m_lr["bleu1"],   "bleu1_sr": m_sr["bleu1"],
                "rouge1_lr": m_lr["rouge1"], "rouge1_sr": m_sr["rouge1"],
                "tfidf_lr": m_lr["tfidf"],   "tfidf_sr": m_sr["tfidf"],
            })

        # Cenários B e C — GT raster (tile PNG da resolução original e reamostrada)
        if gt_path.exists():
            pil_gt = Image.open(gt_path).convert("RGB")
            # Cenário B: GT resolução original
            cap_gt = caption(pil_gt, proc, model, device)
            m_lr_b = compute_metrics(cap_lr, cap_gt)
            m_sr_b = compute_metrics(cap_sr, cap_gt)
            records_gt.append({
                "tile_id": tile_id,
                "caption_lr": cap_lr, "caption_sr": cap_sr, "caption_gt": cap_gt,
                "bleu1_lr": m_lr_b["bleu1"],   "bleu1_sr": m_sr_b["bleu1"],
                "rouge1_lr": m_lr_b["rouge1"], "rouge1_sr": m_sr_b["rouge1"],
                "tfidf_lr": m_lr_b["tfidf"],   "tfidf_sr": m_sr_b["tfidf"],
            })
            # Cenário C: GT reamostrado para resolução da SR (mesmo tile PNG, mas
            # redimensionado para simular a degradação — usa o tile SR como referência
            # de tamanho, igual ao zero-shot-final)
            sr_w, sr_h = pil_sr.size
            pil_gt_rs = pil_gt.resize((sr_w, sr_h), Image.BILINEAR)
            cap_gt_rs = caption(pil_gt_rs, proc, model, device)
            m_lr_c = compute_metrics(cap_lr, cap_gt_rs)
            m_sr_c = compute_metrics(cap_sr, cap_gt_rs)
            records_rs.append({
                "tile_id": tile_id,
                "caption_lr": cap_lr, "caption_sr": cap_sr, "caption_gt_rs": cap_gt_rs,
                "bleu1_lr": m_lr_c["bleu1"],   "bleu1_sr": m_sr_c["bleu1"],
                "rouge1_lr": m_lr_c["rouge1"], "rouge1_sr": m_sr_c["rouge1"],
                "tfidf_lr": m_lr_c["tfidf"],   "tfidf_sr": m_sr_c["tfidf"],
            })

        if i % 50 == 0 or i == len(meta_df):
            print(f"  [{i}/{len(meta_df)}] {tile_id}  "
                  f"(vec={len(records_vec)} gt={len(records_gt)})", flush=True)

    # ── Salvar CSVs ───────────────────────────────────────────────────────────
    if args.resume:
        for csv, records, key_cols in [
            (csv_vec, records_vec, ["tile_id", "classe"]),
            (csv_gt,  records_gt,  ["tile_id"]),
            (csv_rs,  records_rs,  ["tile_id"]),
        ]:
            new_df = pd.DataFrame(records)
            if csv.exists() and len(new_df) > 0:
                old_df = pd.read_csv(csv)
                combined = pd.concat([old_df, new_df], ignore_index=True)
                combined.drop_duplicates(subset=["tile_id"], keep="last", inplace=True)
                combined.to_csv(csv, index=False)
            elif len(new_df) > 0:
                new_df.to_csv(csv, index=False)
    else:
        pd.DataFrame(records_vec).to_csv(csv_vec, index=False)
        pd.DataFrame(records_gt).to_csv(csv_gt,   index=False)
        pd.DataFrame(records_rs).to_csv(csv_rs,   index=False)

    print(f"\nTiles vetorial: {len(records_vec)}")
    print(f"Tiles GT raster: {len(records_gt)}")
    print(f"Tiles GT reamostrado: {len(records_rs)}")

    # ── Gráficos ──────────────────────────────────────────────────────────────
    df_vec = pd.read_csv(csv_vec) if csv_vec.exists() else pd.DataFrame()
    df_gt  = pd.read_csv(csv_gt)  if csv_gt.exists()  else pd.DataFrame()
    df_rs  = pd.read_csv(csv_rs)  if csv_rs.exists()  else pd.DataFrame()

    if len(df_vec) > 0:
        plot_boxplots(df_vec, "GT Vetorial", FIG_DIR / "blip_boxplot_vetorial.png")
        plot_barplot_by_class(df_vec, FIG_DIR / "blip_barplot_por_classe.png")
        print_summary(df_vec, "GT Vetorial")
    if len(df_gt) > 0:
        plot_boxplots(df_gt, "GT Raster 1m", FIG_DIR / "blip_boxplot_raster_gt.png")
        print_summary(df_gt, "GT Raster 1m")
    if len(df_rs) > 0:
        plot_boxplots(df_rs, "GT Raster Reamostrado", FIG_DIR / "blip_boxplot_raster_gt_resampled.png")
        print_summary(df_rs, "GT Raster Reamostrado")

    print(f"\nCSVs em: {OUT_DIR}")
    print(f"Figuras em: {FIG_DIR}")


if __name__ == "__main__":
    main()
