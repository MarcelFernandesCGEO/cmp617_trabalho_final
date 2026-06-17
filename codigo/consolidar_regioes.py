#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolida os resultados de duas regiões geograficamente distintas:

    Região 1 (R1) = resultados/r1_575_tiles/  (575 tiles)
    Região 2 (R2) = resultados/r2_378_tiles/  (378 tiles)

Os esquemas das CSVs são idênticos entre as duas. Este script:
  1. concatena R1 + R2 adicionando a coluna `region` (R1 / R2),
     gravando as CSVs combinadas em resultados/consolidado/;
  2. imprime tabelas-resumo por região (R1, R2) e pooled (R1+R2),
     reproduzindo as tabelas do artigo a partir das CSVs entregues.

Os diretórios padrão apontam para o layout desta entrega. Para rodar sobre
outras execuções, sobrescreva com variáveis de ambiente:
    REGIAO1_DIR, REGIAO2_DIR, CONSOLIDADO_DIR

Uso:
    python consolidar_regioes.py
"""
import os
import sys
import pandas as pd
import numpy as np

# Console do Windows usa cp1252 por padrão — força UTF-8 para os caracteres Δ, →, etc.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
# Layout da entrega (relativo a codigo/): ../resultados/r1_575_tiles e r2_378_tiles.
_DEF_R1 = os.path.normpath(os.path.join(BASE, "..", "resultados", "r1_575_tiles"))
_DEF_R2 = os.path.normpath(os.path.join(BASE, "..", "resultados", "r2_378_tiles"))
_DEF_OUT = os.path.normpath(os.path.join(BASE, "..", "resultados", "consolidado"))
DIR_A = os.environ.get("REGIAO1_DIR", _DEF_R1)
DIR_B = os.environ.get("REGIAO2_DIR", _DEF_R2)
OUT   = os.environ.get("CONSOLIDADO_DIR", _DEF_OUT)
os.makedirs(OUT, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def load_pair(fname):
    """Carrega o mesmo arquivo das duas regiões, adiciona coluna region, concatena."""
    a = pd.read_csv(os.path.join(DIR_A, fname)); a.insert(0, "region", "R1")
    b = pd.read_csv(os.path.join(DIR_B, fname)); b.insert(0, "region", "R2")
    both = pd.concat([a, b], ignore_index=True)
    both.to_csv(os.path.join(OUT, fname), index=False)
    return a, b, both


def pair_stats(df, lr_col, sr_col):
    """Retorna (mean_lr, mean_sr, delta, n_better, n) para um par de colunas LR/SR."""
    v = df[[lr_col, sr_col]].dropna()
    if len(v) == 0:
        return (np.nan, np.nan, np.nan, 0, 0)
    lr, sr = v[lr_col].mean(), v[sr_col].mean()
    nb = int((v[sr_col] > v[lr_col]).sum())
    return (lr, sr, sr - lr, nb, len(v))


def report_metric(title, fname, pairs):
    """pairs = [(label, lr_col, sr_col), ...]"""
    a, b, both = load_pair(fname)
    print(f"\n{'='*78}\n{title}   [{fname}]")
    print(f"{'='*78}")
    print(f"{'métrica':<16}{'região':<8}{'LR':>9}{'SR':>9}{'Δ':>9}   SR>LR")
    for label, lr_col, sr_col in pairs:
        for name, d in (("R1", a), ("R2", b), ("R1+R2", both)):
            lr, sr, dl, nb, n = pair_stats(d, lr_col, sr_col)
            print(f"{label:<16}{name:<8}{lr:>9.3f}{sr:>9.3f}{dl:>+9.3f}   {nb}/{n}")
        print()


def report_clip_by_class(fname):
    """CLIPScore por classe (cenário vetorial) — generic e remote."""
    a, b, both = load_pair(fname)
    print(f"\n{'='*78}\nBLIP CLIPScore por classe (vetorial)   [{fname}]\n{'='*78}")
    for name, d in (("R1", a), ("R2", b), ("R1+R2", both)):
        print(f"\n--- {name} (n={len(d)}) ---")
        g = d.groupby("classe").agg(
            n=("classe", "size"),
            gen_lr=("generic_clip_lr", "mean"), gen_sr=("generic_clip_sr", "mean"),
            rem_lr=("remote_clip_lr", "mean"), rem_sr=("remote_clip_sr", "mean"),
        )
        g["gen_Δ"] = g["gen_sr"] - g["gen_lr"]
        g["rem_Δ"] = g["rem_sr"] - g["rem_lr"]
        print(g.round(3).to_string())


def report_vqa_levels(fname="geochat_vqa_metrics.csv"):
    a, b, both = load_pair(fname)
    print(f"\n{'='*78}\nGeoChat VQA por nível   [{fname}]\n{'='*78}")
    print(f"{'nível':<14}{'região':<8}{'n':>6}{'TF-IDF Δ':>10}{'SR>LR':>9}"
          f"{'Sem. Δ':>10}{'SR>LR':>9}")
    levels = [(1, "semantic"), (2, "structural"), (3, "fine_detail")]
    for lvl, lname in levels:
        for name, d in (("R1", a), ("R2", b), ("R1+R2", both)):
            sub = d[d["level"] == lvl]
            t = sub[["tfidf_lr_gt", "tfidf_sr_gt"]].dropna()
            s = sub[["semantic_lr_gt", "semantic_sr_gt"]].dropna()
            td = t["tfidf_sr_gt"].mean() - t["tfidf_lr_gt"].mean()
            tb = int((t["tfidf_sr_gt"] > t["tfidf_lr_gt"]).sum())
            sd = s["semantic_sr_gt"].mean() - s["semantic_lr_gt"].mean()
            sb = int((s["semantic_sr_gt"] > s["semantic_lr_gt"]).sum())
            print(f"L{lvl} {lname:<10}{name:<8}{len(sub):>6}{td:>+10.3f}{tb:>4}/{len(t):<4}"
                  f"{sd:>+10.3f}{sb:>4}/{len(s):<4}")
        print()

    # q01 MAE
    print("q01_pct_landcover — MAE (pp) vs GT raster:")
    for name, d in (("R1", a), ("R2", b), ("R1+R2", both)):
        q = d[d["question_id"] == "q01_pct_landcover"][["mae_lr_gt", "mae_sr_gt"]].dropna()
        print(f"  {name:<7} LR={q['mae_lr_gt'].mean():6.2f}  SR={q['mae_sr_gt'].mean():6.2f}"
              f"  Δ={q['mae_sr_gt'].mean()-q['mae_lr_gt'].mean():+.2f}  (n={len(q)})")


def report_vqa_by_question(fname="geochat_vqa_metrics.csv"):
    a = pd.read_csv(os.path.join(DIR_A, fname)); a["region"] = "R1"
    b = pd.read_csv(os.path.join(DIR_B, fname)); b["region"] = "R2"
    both = pd.concat([a, b], ignore_index=True)
    print(f"\n{'='*78}\nGeoChat VQA por pergunta — TF-IDF   [{fname}]\n{'='*78}")
    print(f"{'pergunta':<22}{'região':<8}{'LR':>8}{'SR':>8}{'Δ':>9}   SR>LR")
    for qid in sorted(both["question_id"].unique()):
        for name, d in (("R1", a), ("R2", b), ("R1+R2", both)):
            sub = d[d["question_id"] == qid][["tfidf_lr_gt", "tfidf_sr_gt"]].dropna()
            lr, sr = sub["tfidf_lr_gt"].mean(), sub["tfidf_sr_gt"].mean()
            nb = int((sub["tfidf_sr_gt"] > sub["tfidf_lr_gt"]).sum())
            print(f"{qid:<22}{name:<8}{lr:>8.3f}{sr:>8.3f}{sr-lr:>+9.3f}   {nb}/{len(sub)}")
        print()


def report_vqa_vetorial(fname="geochat_vqa_vetorial_per_tile.csv"):
    a, b, both = load_pair(fname)
    classes = ["forest", "agriculture", "urban", "water", "bare_soil"]
    print(f"\n{'='*78}\nVQA × GeoPackage — MAE por classe (pp)   [{fname}]\n{'='*78}")
    for name, d in (("R1", a), ("R2", b), ("R1+R2", both)):
        print(f"\n--- {name} (n={len(d)}) ---")
        print(f"{'classe':<14}{'LR':>8}{'SR':>8}{'Δ(LR-SR)':>10}{'SR>LR':>9}")
        for c in classes:
            lr = d[f"mae_lr_{c}"].mean()
            sr = d[f"mae_sr_{c}"].mean()
            nb = int((d[f"mae_sr_{c}"] < d[f"mae_lr_{c}"]).sum())
            print(f"{c:<14}{lr:>8.2f}{sr:>8.2f}{lr-sr:>+10.2f}{nb:>5}/{len(d)}")
        glob_lr = d["mae_lr"].mean()
        glob_sr = d["mae_sr"].mean()
        glob_gt = d["mae_gt_raster"].mean()
        print(f"{'GLOBAL':<14}{glob_lr:>8.2f}{glob_sr:>8.2f}{glob_lr-glob_sr:>+10.2f}")
        print(f"  GT raster MAE global = {glob_gt:.2f} pp")


def main():
    print(f"Saída consolidada → {OUT}")

    # ---- BLIP NLP ----
    report_metric("BLIP NLP — vetorial (cenário A)", "blip_metrics_vetorial.csv",
                  [("BLEU-1", "bleu1_lr", "bleu1_sr"),
                   ("ROUGE-1", "rouge1_lr", "rouge1_sr"),
                   ("TF-IDF", "tfidf_lr", "tfidf_sr")])
    report_metric("BLIP NLP — GT raster (cenário B)", "blip_metrics_raster_gt.csv",
                  [("BLEU-1", "bleu1_lr", "bleu1_sr"),
                   ("ROUGE-1", "rouge1_lr", "rouge1_sr"),
                   ("TF-IDF", "tfidf_lr", "tfidf_sr")])
    report_metric("BLIP NLP — GT reamostrado (cenário C)", "blip_metrics_raster_gt_resampled.csv",
                  [("BLEU-1", "bleu1_lr", "bleu1_sr"),
                   ("ROUGE-1", "rouge1_lr", "rouge1_sr"),
                   ("TF-IDF", "tfidf_lr", "tfidf_sr")])

    # ---- BLIP CLIPScore ----
    report_metric("BLIP CLIPScore — vetorial (A)", "blip_clipscore_vetorial.csv",
                  [("CLIP genérico", "generic_clip_lr", "generic_clip_sr"),
                   ("RemoteCLIP", "remote_clip_lr", "remote_clip_sr")])
    report_metric("BLIP CLIPScore — GT raster (B)", "blip_clipscore_raster_gt.csv",
                  [("CLIP genérico", "generic_clip_lr", "generic_clip_sr"),
                   ("RemoteCLIP", "remote_clip_lr", "remote_clip_sr")])
    report_metric("BLIP CLIPScore — GT reamostrado (C)", "blip_clipscore_raster_gt_resampled.csv",
                  [("CLIP genérico", "generic_clip_lr", "generic_clip_sr"),
                   ("RemoteCLIP", "remote_clip_lr", "remote_clip_sr")])
    report_clip_by_class("blip_clipscore_vetorial.csv")

    # ---- GeoChat NLP ----
    report_metric("GeoChat NLP — GT raster (B)", "geochat_metrics_raster_gt.csv",
                  [("BLEU-1", "bleu1_lr", "bleu1_sr"),
                   ("ROUGE-1", "rouge1_lr", "rouge1_sr"),
                   ("TF-IDF", "tfidf_lr", "tfidf_sr")])

    # ---- GeoChat CLIPScore híbrido ----
    report_metric("GeoChat CLIPScore híbrido — GT raster (B)", "geochat_clip_raster_gt_hybrid.csv",
                  [("CLIP genérico", "generic_clip_lr", "generic_clip_sr"),
                   ("RemoteCLIP", "remote_clip_lr", "remote_clip_sr")])

    # ---- GeoChat VQA ----
    report_vqa_levels()
    report_vqa_by_question()
    report_vqa_vetorial()

    print(f"\n{'='*78}\nCSVs combinadas gravadas em: {OUT}\n{'='*78}")


if __name__ == "__main__":
    main()
