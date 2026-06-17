#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera as figuras consolidadas multi-região (R1 + R2) para o artigo:

  1. fig_mae_multiregiao.png  — Δ MAE (LR-SR) por classe, R1 vs R2, destacando
     a divergência da classe `water` (SR melhora em R1, piora em R2).
  2. fig_grid_representativo.png — grid LR/SR/GT de tiles representativos das
     duas regiões, incluindo o caso de alucinação de água da R2.
     (Requer os tiles PNG em <regiao>/tiles/{lr,sr,gt}/; como esses arquivos
      são pesados e não acompanham a entrega, esta figura é pulada se eles
      não estiverem presentes — a versão final já está em artigo/figures/.)

Saída → ../artigo/figures/

Diretórios padrão apontam para o layout desta entrega; sobrescreva com as
variáveis de ambiente REGIAO1_DIR e REGIAO2_DIR para outras execuções.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE   = os.path.dirname(os.path.abspath(__file__))
# Layout da entrega (relativo a codigo/): ../resultados/r1_575_tiles e r2_378_tiles.
_DEF_R1 = os.path.normpath(os.path.join(BASE, "..", "resultados", "r1_575_tiles"))
_DEF_R2 = os.path.normpath(os.path.join(BASE, "..", "resultados", "r2_378_tiles"))
DIR_A  = os.environ.get("REGIAO1_DIR", _DEF_R1)
DIR_B  = os.environ.get("REGIAO2_DIR", _DEF_R2)
FIGOUT = os.path.normpath(os.path.join(BASE, "..", "artigo", "figures"))
os.makedirs(FIGOUT, exist_ok=True)

CLASSES = ["forest", "agriculture", "urban", "water", "bare_soil"]
CLABEL  = ["forest", "agriculture", "urban", "water", "bare soil"]


# --------------------------------------------------------------------------
# Figura 1 — Δ MAE por classe, multi-região
# --------------------------------------------------------------------------
def fig_mae_multiregiao():
    a = pd.read_csv(os.path.join(DIR_A, "geochat_vqa_vetorial_per_tile.csv"))
    b = pd.read_csv(os.path.join(DIR_B, "geochat_vqa_vetorial_per_tile.csv"))

    def deltas(df):
        return [df[f"mae_lr_{c}"].mean() - df[f"mae_sr_{c}"].mean() for c in CLASSES]

    dA, dB = deltas(a), deltas(b)
    x = np.arange(len(CLASSES))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    barsA = ax.bar(x - w/2, dA, w, label="Região 1 (n=575)", color="#2c7fb8")
    barsB = ax.bar(x + w/2, dB, w, label="Região 2 (n=378)", color="#de8a3a")

    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLABEL)
    ax.set_ylabel(r"$\Delta$ MAE (LR$-$SR), pp  —  positivo = SR melhor")
    ax.set_title("Ganho da SR na estimativa de cobertura por classe e região")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    for bars in (barsA, barsB):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:+.1f}", (bar.get_x() + bar.get_width()/2, h),
                        ha="center", va="bottom" if h >= 0 else "top",
                        fontsize=7, xytext=(0, 2 if h >= 0 else -2),
                        textcoords="offset points")

    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(FIGOUT, "fig_mae_multiregiao.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("ok:", out)


# --------------------------------------------------------------------------
# Figura 2 — grid representativo LR/SR/GT
# --------------------------------------------------------------------------
def _patch(region_dir, sub, tile):
    p = os.path.join(region_dir, "tiles", sub, f"{tile}.png")
    return Image.open(p).convert("RGB")


def fig_grid_representativo():
    # (rótulo, diretório da região, tile_id)
    rows = [
        ("R1 · water — SR recupera o corpo d'água",      DIR_A, "tile_0028"),
        ("R1 · bare soil — SR corrige superestimativa",  DIR_A, "tile_0230"),
        ("R2 · agriculture — SR aproxima do GT",         DIR_B, "tile_0040"),
        ("R2 · water — SR superestima água (rara)",      DIR_B, "tile_0056"),
    ]
    cols = ["LR (~11,5 m/px)", "SR (~2,4 m/px)", "GT (ortofoto)"]
    subs = ["lr", "sr", "gt"]

    # Os tiles PNG são pesados e não acompanham a entrega: pular se ausentes.
    faltando = [t for _, d, t in rows
                if not os.path.exists(os.path.join(d, "tiles", "lr", f"{t}.png"))]
    if faltando:
        print("skip: fig_grid_representativo (tiles PNG ausentes na entrega; "
              "a figura final já está em artigo/figures/fig_grid_representativo.png)")
        return

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(6.4, 2.15 * n))
    for r, (label, rdir, tile) in enumerate(rows):
        for c, sub in enumerate(subs):
            ax = axes[r, c]
            ax.imshow(_patch(rdir, sub, tile))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
        # rótulo da linha à esquerda (neutro, sem destaque de cor)
        axes[r, 0].set_ylabel(label, fontsize=7.6, color="#222", rotation=90,
                              labelpad=6, va="center")
    fig.suptitle("Tiles representativos das duas regiões — LR / SR / GT",
                 fontsize=10, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(FIGOUT, "fig_grid_representativo.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("ok:", out)


if __name__ == "__main__":
    fig_mae_multiregiao()
    fig_grid_representativo()
    print("Figuras gravadas em:", FIGOUT)
