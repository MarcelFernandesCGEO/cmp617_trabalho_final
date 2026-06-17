"""
blip_clipscore_metrics.py
=========================
CLIPScore com BLIP para 575 tiles — Cenários A (GT vetorial), B (GT raster 1m)
e C (GT raster reamostrado para resolução da SR).

Lê tiles PNG já gerados por prepare_data.py e tiles_metadata.csv para bboxes.

Dois modelos CLIP:
  - CLIP genérico : openai/clip-vit-base-patch32
  - RemoteCLIP    : chendelong/RemoteCLIP (open_clip + pesos locais)

CLIPScore = 2.5 * max(cos(a, b), 0)

Saídas:
  results/blip_clipscore_vetorial.csv
  results/blip_clipscore_raster_gt.csv
  results/blip_clipscore_raster_gt_resampled.csv
  results/figures/blip_clip_*.png

Uso:
  python blip_clipscore_metrics.py
  python blip_clipscore_metrics.py --max-tiles 10
  python blip_clipscore_metrics.py --resume
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
import torch.nn.functional as F
from transformers.utils import import_utils as _iu
import transformers.modeling_utils as _mu
_iu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None
from transformers import CLIPProcessor, CLIPModel
import open_clip
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

CLIP_W = 2.5

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
# CLIPScore
# ---------------------------------------------------------------------------
def clipscore(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    cos = F.cosine_similarity(emb_a, emb_b, dim=-1).item()
    return float(CLIP_W * max(cos, 0.0))

# ---------------------------------------------------------------------------
# CLIP genérico
# ---------------------------------------------------------------------------
class GenericCLIP:
    def __init__(self, device: str):
        self.device = device
        print("  Carregando CLIP genérico (local_files_only)...")
        from huggingface_hub import snapshot_download
        clip_dir = snapshot_download(
            "openai/clip-vit-base-patch32", local_files_only=True
        )
        self.proc  = CLIPProcessor.from_pretrained(clip_dir)
        self.model = CLIPModel.from_pretrained(clip_dir).to(device)
        self.model.eval()

    @torch.no_grad()
    def embed_image(self, pil_img: Image.Image) -> torch.Tensor:
        inputs = self.proc(images=pil_img, return_tensors="pt").to(self.device)
        out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        feat = self.model.visual_projection(out.pooler_output)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        inputs = self.proc(text=[text], return_tensors="pt",
                           padding=True, truncation=True).to(self.device)
        out = self.model.text_model(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"])
        feat = self.model.text_projection(out.pooler_output)
        return F.normalize(feat, dim=-1)

# ---------------------------------------------------------------------------
# RemoteCLIP
# ---------------------------------------------------------------------------
class RemoteCLIPModel:
    def __init__(self, device: str):
        self.device = device
        print("  Carregando RemoteCLIP (local_files_only)...")
        # Procura pesos locais
        ckpt_candidates = list(BASE.rglob("RemoteCLIP-ViT-B-32.pt"))
        if not ckpt_candidates:
            # Tenta via huggingface_hub com local_files_only
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id="chendelong/RemoteCLIP",
                filename="RemoteCLIP-ViT-B-32.pt",
                local_files_only=True,
            )
        else:
            ckpt_path = str(ckpt_candidates[0])
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=None
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def embed_image(self, pil_img: Image.Image) -> torch.Tensor:
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(tensor)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        feat = self.model.encode_text(tokens)
        return F.normalize(feat, dim=-1)

# ---------------------------------------------------------------------------
# GeoPackage
# ---------------------------------------------------------------------------
def load_gpkg(gpkg_path: Path) -> dict:
    """Resolve variantes de nome com/sem sufixo .shp."""
    norm_map = {k.removesuffix(".shp").lower(): (k, v)
                for k, v in LAYER_CLASS_MAP.items()}
    gdfs = {}
    for name, _ in pyogrio.list_layers(str(gpkg_path)):
        key = name.removesuffix(".shp").lower()
        if key in norm_map:
            gdf = gpd.read_file(str(gpkg_path), layer=name).to_crs("EPSG:3857")
            orig_key, _ = norm_map[key]
            gdfs[orig_key] = gdf
    return gdfs

def dominant_class(bbox: tuple, gdfs: dict) -> str | None:
    from shapely.geometry import box as shapely_box
    tile_geom = shapely_box(*bbox)
    tile_area = tile_geom.area
    seen_classes = {}
    for layer_name, gdf in gdfs.items():
        cls = LAYER_CLASS_MAP[layer_name]
        try:
            area = gdf.geometry.intersection(tile_geom).area.sum()
        except Exception:
            area = 0.0
        seen_classes[cls] = seen_classes.get(cls, 0.0) + area
    best_cls, best_area = max(seen_classes.items(), key=lambda x: x[1], default=(None, 0.0))
    if best_area < 0.05 * tile_area:
        return None
    return best_cls

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
MODELS = ["generic_clip", "remote_clip"]
MODEL_LABELS = {"generic_clip": "CLIP genérico", "remote_clip": "RemoteCLIP"}

def plot_boxplots(df: pd.DataFrame, scenario: str, fname: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"BLIP CLIPScore — {scenario}", fontsize=13)
    for ax, m in zip(axes, MODELS):
        lc, sc = f"{m}_lr", f"{m}_sr"
        if lc not in df.columns:
            ax.set_visible(False)
            continue
        bp = ax.boxplot([df[lc].dropna().values, df[sc].dropna().values],
                        labels=["LR", "SR"], patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("#5dade2")
        bp["boxes"][1].set_facecolor("#f39c12")
        ax.set_title(MODEL_LABELS[m])
        ax.set_ylabel("CLIPScore")
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()

def plot_barplot_by_class(df: pd.DataFrame, fname: Path):
    classes = df["classe"].dropna().unique()
    if len(classes) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(classes) * 3), 5))
    fig.suptitle("BLIP CLIPScore por Classe — GT Vetorial", fontsize=13)
    for ax, m in zip(axes, MODELS):
        lc, sc = f"{m}_lr", f"{m}_sr"
        if lc not in df.columns:
            ax.set_visible(False)
            continue
        x = np.arange(len(classes))
        lr_m = [df[df["classe"] == c][lc].mean() for c in classes]
        sr_m = [df[df["classe"] == c][sc].mean() for c in classes]
        ax.bar(x - 0.175, lr_m, 0.35, label="LR", color="#5dade2")
        ax.bar(x + 0.175, sr_m, 0.35, label="SR", color="#f39c12")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_title(MODEL_LABELS[m])
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()

def plot_resolution_comparison(df_gt: pd.DataFrame, df_rs: pd.DataFrame, fname: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Impacto da Resolução do GT no CLIPScore\nGT 1m vs GT reamostrado", fontsize=11)
    for ax, m in zip(axes, MODELS):
        sc = f"{m}_sr"; lc = f"{m}_lr"
        if sc not in df_gt.columns:
            ax.set_visible(False)
            continue
        vals = {
            "LR vs GT 1m" : df_gt[lc].mean(),
            "SR vs GT 1m" : df_gt[sc].mean(),
            "LR vs GT rs" : df_rs[lc].mean(),
            "SR vs GT rs" : df_rs[sc].mean(),
        }
        colors = ["#5dade2", "#f39c12", "#85c1e9", "#f8c471"]
        bars = ax.bar(list(vals.keys()), list(vals.values()), color=colors)
        for bar, val in zip(bars, vals.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(MODEL_LABELS[m])
        ax.set_ylabel("CLIPScore médio")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()

def print_summary(df: pd.DataFrame, scenario: str):
    n = len(df)
    print(f"\n{'='*65}")
    print(f"Resumo CLIPScore — {scenario}  (n={n})")
    print(f"{'='*65}")
    for m in MODELS:
        lc, sc = f"{m}_lr", f"{m}_sr"
        if lc not in df.columns:
            continue
        lr, sr = df[lc].dropna(), df[sc].dropna()
        wins = (df[sc] > df[lc]).sum()
        print(f"  {MODEL_LABELS[m]:20s}  LR={lr.mean():.4f}  SR={sr.mean():.4f}  "
              f"Δ={sr.mean()-lr.mean():+.4f}  SR>LR={wins}/{n}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    for p, name in [(META_CSV, "tiles_metadata.csv"), (GPKG_PATH, "GeoPackage")]:
        if not p.exists():
            print(f"ERRO: {name} não encontrado em {p}")
            sys.exit(1)

    meta_df = pd.read_csv(META_CSV)
    if args.max_tiles:
        meta_df = meta_df.head(args.max_tiles)

    csv_vec = OUT_DIR / "blip_clipscore_vetorial.csv"
    csv_gt  = OUT_DIR / "blip_clipscore_raster_gt.csv"
    csv_rs  = OUT_DIR / "blip_clipscore_raster_gt_resampled.csv"

    done_ids = set()
    if args.resume:
        for csv in [csv_vec, csv_gt, csv_rs]:
            if csv.exists():
                done_ids.update(pd.read_csv(csv)["tile_id"].tolist())
        print(f"  Resume: {len(done_ids)} tile_ids já processados")

    print("Carregando GeoPackage...")
    gdfs = load_gpkg(GPKG_PATH)
    print(f"  {len(gdfs)} camadas carregadas")

    print("Carregando modelos CLIP...")
    clip = GenericCLIP(device)
    remote = None
    try:
        remote = RemoteCLIPModel(device)
    except Exception as e:
        print(f"  AVISO: RemoteCLIP não carregado — {e}")

    # Pré-computar embeddings de texto por classe
    text_emb_g = {cls: clip.embed_text(cls) for cls in set(LAYER_CLASS_MAP.values())}
    text_emb_r = ({cls: remote.embed_text(cls) for cls in set(LAYER_CLASS_MAP.values())}
                  if remote else {})

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

        emb_lr_g = clip.embed_image(pil_lr)
        emb_sr_g = clip.embed_image(pil_sr)
        emb_lr_r = remote.embed_image(pil_lr) if remote else None
        emb_sr_r = remote.embed_image(pil_sr) if remote else None

        # Cenário A — GT vetorial
        bbox = (row.x0, row.y0, row.x1, row.y1)
        cls = dominant_class(bbox, gdfs)
        if cls is not None:
            rec = {"tile_id": tile_id, "classe": cls,
                   "generic_clip_lr": clipscore(emb_lr_g, text_emb_g[cls]),
                   "generic_clip_sr": clipscore(emb_sr_g, text_emb_g[cls])}
            if remote:
                rec["remote_clip_lr"] = clipscore(emb_lr_r, text_emb_r[cls])
                rec["remote_clip_sr"] = clipscore(emb_sr_r, text_emb_r[cls])
            records_vec.append(rec)

        # Cenários B e C — GT raster
        if gt_path.exists():
            pil_gt = Image.open(gt_path).convert("RGB")
            emb_gt_g = clip.embed_image(pil_gt)

            # Cenário B
            rec_b = {"tile_id": tile_id,
                     "generic_clip_lr": clipscore(emb_lr_g, emb_gt_g),
                     "generic_clip_sr": clipscore(emb_sr_g, emb_gt_g)}
            if remote:
                emb_gt_r = remote.embed_image(pil_gt)
                rec_b["remote_clip_lr"] = clipscore(emb_lr_r, emb_gt_r)
                rec_b["remote_clip_sr"] = clipscore(emb_sr_r, emb_gt_r)
            records_gt.append(rec_b)

            # Cenário C — GT reamostrado para resolução da SR
            sr_w, sr_h = pil_sr.size
            pil_gt_rs  = pil_gt.resize((sr_w, sr_h), Image.BILINEAR)
            emb_gt_rs_g = clip.embed_image(pil_gt_rs)
            rec_c = {"tile_id": tile_id,
                     "generic_clip_lr": clipscore(emb_lr_g, emb_gt_rs_g),
                     "generic_clip_sr": clipscore(emb_sr_g, emb_gt_rs_g)}
            if remote:
                emb_gt_rs_r = remote.embed_image(pil_gt_rs)
                rec_c["remote_clip_lr"] = clipscore(emb_lr_r, emb_gt_rs_r)
                rec_c["remote_clip_sr"] = clipscore(emb_sr_r, emb_gt_rs_r)
            records_rs.append(rec_c)

        if i % 50 == 0 or i == len(meta_df):
            print(f"  [{i}/{len(meta_df)}] {tile_id}  "
                  f"(vec={len(records_vec)} gt={len(records_gt)})", flush=True)

    # ── Salvar CSVs ───────────────────────────────────────────────────────────
    if args.resume:
        for csv, records in [(csv_vec, records_vec), (csv_gt, records_gt), (csv_rs, records_rs)]:
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
        plot_boxplots(df_vec, "GT Vetorial", FIG_DIR / "blip_clip_boxplot_vetorial.png")
        plot_barplot_by_class(df_vec, FIG_DIR / "blip_clip_barplot_por_classe.png")
        print_summary(df_vec, "GT Vetorial")
    if len(df_gt) > 0:
        plot_boxplots(df_gt, "GT Raster 1m", FIG_DIR / "blip_clip_boxplot_raster_gt.png")
        print_summary(df_gt, "GT Raster 1m")
    if len(df_rs) > 0:
        plot_boxplots(df_rs, "GT Raster Reamostrado", FIG_DIR / "blip_clip_boxplot_raster_gt_resampled.png")
        print_summary(df_rs, "GT Raster Reamostrado")
    if len(df_gt) > 0 and len(df_rs) > 0:
        plot_resolution_comparison(df_gt, df_rs, FIG_DIR / "blip_clip_gt_resolution_comparison.png")

    print(f"\nCSVs em: {OUT_DIR}")
    print(f"Figuras em: {FIG_DIR}")


if __name__ == "__main__":
    main()
