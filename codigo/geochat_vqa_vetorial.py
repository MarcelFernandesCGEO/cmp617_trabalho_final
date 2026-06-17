"""
geochat_vqa_vetorial.py
=======================
Cruzamento entre respostas VQA (q01_pct_landcover) e ground truth vetorial
(GeoPackage) para avaliação de cobertura do solo por tile.

Para cada tile:
  1. Intersecta as 7 camadas do GeoPackage com o bbox do tile
  2. Calcula a % real de cada classe VQA (forest, agriculture, urban, water, bare soil)
  3. Compara com as estimativas de LR, SR e GT da pergunta q01
  4. Métricas por tile: MAE por classe e MAE global

Saídas:
  results/geochat_vqa_vetorial_per_tile.csv  ← MAE por tile e classe
  results/geochat_vqa_vetorial_summary.csv   ← resumo por classe (média LR vs SR vs GT)
  results/figures/vqa_vetorial_*.png         ← gráficos

Uso:
  python geochat_vqa_vetorial.py
  python geochat_vqa_vetorial.py --max-tiles 42
"""

import os
import sys
import argparse
import re
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import box as shapely_box

warnings.filterwarnings("ignore")
logging.getLogger("geopandas").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE        = Path(__file__).parent
GPKG_PATH   = BASE / "data" / "vetores_cobertura_solo.gpkg"
TILES_DIR   = BASE / "results" / "tiles"
OUT_DIR     = BASE / "results"
FIG_DIR     = BASE / "results" / "figures"
os.makedirs(FIG_DIR, exist_ok=True)

ANSWERS_CSV = OUT_DIR / "geochat_vqa_answers.csv"
META_CSV    = TILES_DIR / "tiles_metadata.csv"

# ---------------------------------------------------------------------------
# Mapeamento camada GeoPackage → classe VQA
# ---------------------------------------------------------------------------
LAYER_TO_VQA = {
    "VEG_Floresta_A.shp":               "forest",
    "Veg_Cultivada_A.shp":              "agriculture",
    "VEG_Campo_A.shp":                  "agriculture",
    "HID_Massa_Dagua_A":                "water",
    "VEG_Brejo_Pantano_A":              "bare soil",
    "REL_Terreno_Exposto_A":            "bare soil",
    "LML_Area_Densamente_Edificada_A":  "urban",
}

VQA_CLASSES = ["forest", "agriculture", "urban", "water", "bare soil"]

# ---------------------------------------------------------------------------
# Parser de respostas q01 (mesmo padrão do geochat_vqa_metrics.py)
# ---------------------------------------------------------------------------
def parse_pct(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    result = {}
    for key in VQA_CLASSES:
        m = re.search(rf"{key}\s+(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
        if m:
            result[key] = float(m.group(1))
    return result if len(result) >= 3 else None


def mae_vs_gt(pred: dict | None, gt_pct: dict) -> float | None:
    """MAE entre predição do modelo e GT vetorial (pp)."""
    if pred is None:
        return None
    errors = [abs(pred.get(k, 0.0) - gt_pct.get(k, 0.0)) for k in VQA_CLASSES]
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Ground truth vetorial por tile
# ---------------------------------------------------------------------------
def load_gpkg(gpkg_path: Path) -> dict:
    """Carrega todas as camadas do GeoPackage em EPSG:3857."""
    import geopandas as gpd
    import pyogrio
    layers_info = pyogrio.list_layers(str(gpkg_path))
    gdfs = {}
    for name, _ in layers_info:
        gdf = gpd.read_file(str(gpkg_path), layer=name).to_crs("EPSG:3857")
        gdfs[name] = gdf
    return gdfs


def tile_gt_pct(bbox: tuple, gdfs: dict) -> dict:
    """
    Calcula o % real de cada classe VQA dentro do tile.
    bbox = (x0, y0, x1, y1) em EPSG:3857.
    Retorna dict {classe_vqa: pct} onde pct ∈ [0, 100].
    """
    tile_geom  = shapely_box(*bbox)
    tile_area  = tile_geom.area

    class_area = {c: 0.0 for c in VQA_CLASSES}

    for layer_name, vqa_class in LAYER_TO_VQA.items():
        gdf = gdfs.get(layer_name)
        if gdf is None or gdf.empty:
            continue
        try:
            inter = gdf.geometry.intersection(tile_geom)
            class_area[vqa_class] += inter.area.sum()
        except Exception:
            pass

    # normaliza pela área mapeada total dentro do tile
    # (pode ser < tile_area nas bordas, mas o usuário garante cobertura ~100%)
    mapped = sum(class_area.values())
    if mapped < 1.0:
        return {c: 0.0 for c in VQA_CLASSES}

    return {c: round(class_area[c] / mapped * 100, 2) for c in VQA_CLASSES}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--answers-csv", type=Path, default=ANSWERS_CSV,
                        help="Caminho para geochat_vqa_answers.csv")
    parser.add_argument("--meta-csv", type=Path, default=META_CSV,
                        help="Caminho para tiles_metadata.csv")
    args = parser.parse_args()

    # ── Verificações ─────────────────────────────────────────────────────────
    answers_path = args.answers_csv
    meta_path    = args.meta_csv

    for path, name in [(GPKG_PATH, "GeoPackage"), (answers_path, "geochat_vqa_answers.csv"),
                       (meta_path, "tiles_metadata.csv")]:
        if not path.exists():
            print(f"ERRO: {name} não encontrado em {path}")
            sys.exit(1)

    # ── Carregar dados ────────────────────────────────────────────────────────
    print("Carregando GeoPackage...")
    gdfs = load_gpkg(GPKG_PATH)
    print(f"  {len(gdfs)} camadas carregadas")

    print("Carregando metadados de tiles...")
    meta_df = pd.read_csv(META_CSV)
    if args.max_tiles:
        meta_df = meta_df.head(args.max_tiles)
    print(f"  {len(meta_df)} tiles")

    print("Carregando respostas VQA...")
    answers_df = pd.read_csv(ANSWERS_CSV)
    q01 = answers_df[answers_df["question_id"] == "q01_pct_landcover"].copy()
    q01 = q01.set_index("tile_id")
    print(f"  {len(q01)} respostas q01")

    # ── Processar tiles ───────────────────────────────────────────────────────
    print(f"\nProcessando {len(meta_df)} tiles...")
    records = []

    for i, row in enumerate(meta_df.itertuples(), 1):
        tile_id = row.tile_id
        bbox    = (row.x0, row.y0, row.x1, row.y1)

        # GT vetorial real
        gt_pct = tile_gt_pct(bbox, gdfs)

        # Respostas do modelo
        if tile_id not in q01.index:
            continue
        r = q01.loc[tile_id]

        pred_lr  = parse_pct(r.get("answer_lr",  ""))
        pred_sr  = parse_pct(r.get("answer_sr",  ""))
        pred_gtr = parse_pct(r.get("answer_gt",  ""))  # GT raster (câmera aérea)

        mae_lr  = mae_vs_gt(pred_lr,  gt_pct)
        mae_sr  = mae_vs_gt(pred_sr,  gt_pct)
        mae_gtr = mae_vs_gt(pred_gtr, gt_pct)

        rec = {
            "tile_id": tile_id,
            "mae_lr":  mae_lr,
            "mae_sr":  mae_sr,
            "mae_gt_raster": mae_gtr,
            "refused_lr": pred_lr is None,
            "refused_sr": pred_sr is None,
            "refused_gt_raster": pred_gtr is None,
        }

        # % GT vetorial por classe
        for c in VQA_CLASSES:
            rec[f"gt_{c.replace(' ', '_')}"] = gt_pct[c]

        # % predito por classe (LR e SR)
        for c in VQA_CLASSES:
            ck = c.replace(" ", "_")
            rec[f"lr_{ck}"] = pred_lr.get(c, None) if pred_lr else None
            rec[f"sr_{ck}"] = pred_sr.get(c, None) if pred_sr else None

        # MAE por classe (LR e SR)
        for c in VQA_CLASSES:
            ck = c.replace(" ", "_")
            rec[f"mae_lr_{ck}"]  = abs(pred_lr.get(c, 0) - gt_pct[c]) if pred_lr else None
            rec[f"mae_sr_{ck}"]  = abs(pred_sr.get(c, 0) - gt_pct[c]) if pred_sr else None

        records.append(rec)

        if i % 10 == 0 or i == len(meta_df):
            print(f"  {i}/{len(meta_df)} tiles processados", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "geochat_vqa_vetorial_per_tile.csv", index=False)

    # ── Resumo ────────────────────────────────────────────────────────────────
    n_total  = len(df)
    n_valid_lr  = df["mae_lr"].notna().sum()
    n_valid_sr  = df["mae_sr"].notna().sum()
    n_valid_gtr = df["mae_gt_raster"].notna().sum()

    print(f"\n{'='*60}")
    print(f"Resumo VQA × GT Vetorial — {n_total} tiles")
    print(f"{'='*60}")
    print(f"  Respostas válidas: LR={n_valid_lr}  SR={n_valid_sr}  GT_raster={n_valid_gtr}")
    print()

    # MAE global
    mae_lr_mean  = df["mae_lr"].mean()
    mae_sr_mean  = df["mae_sr"].mean()
    mae_gtr_mean = df["mae_gt_raster"].mean()
    delta        = mae_lr_mean - mae_sr_mean  # positivo = SR melhor
    print(f"  MAE global vs GT vetorial (pp):")
    print(f"    LR      : {mae_lr_mean:.2f} pp")
    print(f"    SR      : {mae_sr_mean:.2f} pp")
    print(f"    GT raster: {mae_gtr_mean:.2f} pp")
    print(f"    Δ (LR−SR): {delta:+.2f} pp  {'← SR melhor' if delta > 0 else '← LR melhor'}")

    # MAE por classe
    print(f"\n  MAE por classe (SR melhor = Δ > 0):")
    summary_rows = []
    for c in VQA_CLASSES:
        ck = c.replace(" ", "_")
        col_lr = f"mae_lr_{ck}"
        col_sr = f"mae_sr_{ck}"
        if col_lr not in df.columns:
            continue
        m_lr = df[col_lr].mean()
        m_sr = df[col_sr].mean()
        d = m_lr - m_sr
        sr_better = (df[col_lr] > df[col_sr]).sum()
        print(f"    {c:<15}: LR={m_lr:.1f}pp  SR={m_sr:.1f}pp  Δ={d:+.1f}pp  SR>LR: {sr_better}/{n_total}")
        summary_rows.append({
            "classe": c,
            "mae_lr_pp": round(m_lr, 2),
            "mae_sr_pp": round(m_sr, 2),
            "delta_pp":  round(d, 2),
            "sr_better_n": sr_better,
            "n_tiles": n_total,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "geochat_vqa_vetorial_summary.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_mae_por_classe(summary_df, FIG_DIR / "vqa_vetorial_mae_por_classe.png")
    _plot_scatter_lr_sr(df, FIG_DIR / "vqa_vetorial_scatter.png")

    print(f"\nCSVs: {OUT_DIR}/geochat_vqa_vetorial_*.csv")
    print(f"Figuras: {FIG_DIR}/vqa_vetorial_*.png")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_mae_por_classe(summary_df: pd.DataFrame, fname: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(summary_df))
    w = 0.35
    ax.bar(x - w/2, summary_df["mae_lr_pp"], w, label="LR", color="#e07070")
    ax.bar(x + w/2, summary_df["mae_sr_pp"], w, label="SR", color="#70a8e0")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["classe"], rotation=20, ha="right")
    ax.set_ylabel("MAE vs GT vetorial (pp)")
    ax.set_title("Erro de estimativa de cobertura do solo por classe\nLR vs SR comparado ao GeoPackage")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()


def _plot_scatter_lr_sr(df: pd.DataFrame, fname: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label, color in [
        (axes[0], "mae_lr", "LR", "#e07070"),
        (axes[1], "mae_sr", "SR", "#70a8e0"),
    ]:
        valid = df[col].dropna()
        ax.hist(valid, bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(valid.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"média={valid.mean():.1f}pp")
        ax.set_title(f"MAE global — {label}")
        ax.set_xlabel("MAE vs GT vetorial (pp)")
        ax.set_ylabel("Tiles")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle("Distribuição do erro de estimativa de cobertura do solo", y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
