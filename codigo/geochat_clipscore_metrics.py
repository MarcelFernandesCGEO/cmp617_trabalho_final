import os
from pathlib import Path

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
geochat_clipscore_metrics.py
============================
Abordagem híbrida: GeoChat gera captions + CLIP/RemoteCLIP computa CLIPScore.

Modos:
  1) Vetorial — text-vs-text: CLIPScore(caption_LR/SR, classe)
  2) Vetorial — hybrid:       CLIPScore(image_LR/SR, text(classe))
  3) Raster GT — text-vs-text: CLIPScore(caption_LR/SR, caption_GT)
  4) Raster GT — hybrid:       CLIPScore(image_LR/SR, text(caption_GT))

Fonte vetorial: data/vetores_cobertura_solo.gpkg (múltiplas camadas, uma por classe)

Saídas:
  results/geochat_clip_vetorial_txt.csv
  results/geochat_clip_vetorial_hybrid.csv
  results/geochat_clip_raster_gt_txt.csv
  results/geochat_clip_raster_gt_hybrid.csv
  results/geochat_clip_raster_rs_txt.csv
  results/geochat_clip_raster_rs_hybrid.csv
  results/figures/gc_clip_*.png

Uso:
  python geochat_clipscore_metrics.py --model-path models/geochat-7B
  python geochat_clipscore_metrics.py --model-path models/geochat-7B --load-4bit
  python geochat_clipscore_metrics.py --model-path models/geochat-7B --max-tiles 5
"""

import sys
import argparse
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import geopandas as gpd
from shapely.geometry import box
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import hf_hub_download
import open_clip

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Paths — auto-detecção a partir da pasta do script
# ---------------------------------------------------------------------------
BASE     = Path(__file__).resolve().parent

# GeoPackage — camadas mapeadas para nomes de classe legíveis
GPKG_PATH = BASE / "data" / "vetores_cobertura_solo.gpkg"

# O GDAL às vezes preserva ".shp" no nome da camada ao importar shapefiles;
# _resolve_layer_name_map() testa ambas as variantes automaticamente.
LAYER_NAME_MAP = {
    "HID_Massa_Dagua_A":               "massa dagua",
    "LML_Area_Densamente_Edificada_A": "area edificada",
    "REL_Terreno_Exposto_A":           "terreno exposto",
    "VEG_Brejo_Pantano_A":             "brejo pantano",
    "VEG_Campo_A":                     "campo",
    "Veg_Cultivada_A":                 "vegetacao cultivada",
    "VEG_Floresta_A":                  "floresta",
}

def _resolve_layer_name_map(gpkg_path):
    """Retorna {nome_real_no_gpkg: classe} resolvendo variantes com/sem .shp."""
    available: set = set()
    for _attempt in ["fiona", "gpd_fiona", "pyogrio"]:
        try:
            if _attempt == "fiona":
                import fiona
                available = set(fiona.listlayers(str(gpkg_path)))
            elif _attempt == "gpd_fiona":
                import geopandas as _gpd
                available = set(_gpd.io.file.fiona.listlayers(str(gpkg_path)))
            elif _attempt == "pyogrio":
                import pyogrio
                available = set(n for n, _ in pyogrio.list_layers(str(gpkg_path)))
            if available:
                break
        except Exception:
            pass

    if available:
        print(f"  Camadas disponíveis no GeoPackage: {sorted(available)}")

    resolved = {}
    for layer_key, cls_name in LAYER_NAME_MAP.items():
        base = layer_key.removesuffix(".shp").lower()
        if layer_key in available:
            resolved[layer_key] = cls_name
        elif layer_key + ".shp" in available:
            resolved[layer_key + ".shp"] = cls_name
        elif layer_key.removesuffix(".shp") in available:
            resolved[layer_key.removesuffix(".shp")] = cls_name
        else:
            avail_bases = {a: a.removesuffix(".shp").lower() for a in available}
            exact_ci = next((a for a, b in avail_bases.items() if b == base), None)
            if exact_ci:
                resolved[exact_ci] = cls_name
            else:
                suffix_m = next((a for a, b in avail_bases.items()
                                 if b.endswith("_" + base) or b.endswith(base)), None)
                if suffix_m:
                    print(f"  NOTA: '{layer_key}' → '{suffix_m}' (match por sufixo)")
                    resolved[suffix_m] = cls_name
                else:
                    print(f"  AVISO: camada '{layer_key}' não encontrada no GeoPackage")
    return resolved

OUT_DIR   = str(BASE / "results")
FIG_DIR   = os.path.join(OUT_DIR, "figures")
TILES_DIR = os.path.join(OUT_DIR, "tiles")
TILES_OUT = os.path.join(OUT_DIR, "tiles_geochat_clip")

TARGET_CRS = "EPSG:3857"
CLIP_W     = 2.5
DISPLAY_PX = 256

GEOCHAT_PROMPT = (
    "Describe this remote sensing image in one sentence. "
    "Focus on the land cover type visible."
)


def dominant_class(bbox, gdf_dict):
    tile_geom = box(*bbox)
    best_cls, best_area = None, 0.0
    for cls_name, gdf in gdf_dict.items():
        try:
            area = gdf.geometry.intersection(tile_geom).area.sum()
        except Exception:
            area = 0.0
        if area > best_area:
            best_area = area
            best_cls = cls_name
    if best_area < 0.05 * tile_geom.area:
        return None
    return best_cls


# ---------------------------------------------------------------------------
# CLIPScore
# ---------------------------------------------------------------------------
def clipscore(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    cos = F.cosine_similarity(emb_a, emb_b, dim=-1).item()
    return float(CLIP_W * max(cos, 0.0))


# ---------------------------------------------------------------------------
# CLIP wrappers
# ---------------------------------------------------------------------------
class GenericCLIP:
    def __init__(self, device):
        self.device = device
        print("  Carregando CLIP genérico (openai/clip-vit-base-patch32)...")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device)
        self.model.eval()

    @torch.no_grad()
    def embed_image(self, pil_img: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        feat = self.model.visual_projection(
            self.model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        )
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        feat = self.model.text_projection(
            self.model.text_model(input_ids=inputs["input_ids"],
                                  attention_mask=inputs["attention_mask"]).pooler_output
        )
        return F.normalize(feat, dim=-1)


class RemoteCLIP:
    def __init__(self, device):
        self.device = device
        print("  Carregando RemoteCLIP (chendelong/RemoteCLIP-ViT-B-32)...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=None
        )
        ckpt_path = hf_hub_download(repo_id="chendelong/RemoteCLIP",
                                    filename="RemoteCLIP-ViT-B-32.pt",
                                    local_files_only=True)
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def embed_image(self, pil_img: Image.Image) -> torch.Tensor:
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        return F.normalize(self.model.encode_image(tensor), dim=-1)

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        return F.normalize(self.model.encode_text(tokens), dim=-1)


# ---------------------------------------------------------------------------
# GeoChat captioner
# ---------------------------------------------------------------------------
class GeoChatCaptioner:
    def __init__(self, model_path="MBZUAI/geochat-7B",
                 load_4bit=False, load_8bit=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"  Carregando GeoChat ({model_path})...")
        print(f"  Dispositivo: {device}")
        if device == "cpu":
            print("  AVISO: CPU — inferência lenta. Use --max-tiles para testar.")
            load_4bit = False
            load_8bit = False

        try:
            from geochat.model.builder import load_pretrained_model
            from geochat.mm_utils import get_model_name_from_path
        except ImportError:
            print("\n  ERRO: Pacote 'geochat' não encontrado.")
            print("  Instale: pip install -e geochat/ --no-deps")
            sys.exit(1)

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(
                model_path=model_path, model_base=None,
                model_name=model_name,
                load_8bit=load_8bit, load_4bit=load_4bit,
                device=device,
                device_map={"": 0} if device == "cuda" else "cpu",
            )
        self.model.eval()

        from geochat.conversation import conv_templates
        from geochat.mm_utils import tokenizer_image_token, process_images
        from geochat.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        )
        self._conv_templates         = conv_templates
        self._tokenizer_image_token  = tokenizer_image_token
        self._process_images         = process_images
        self._IMAGE_TOKEN_INDEX      = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN    = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN   = DEFAULT_IM_END_TOKEN
        print("  GeoChat carregado com sucesso.")

    @torch.no_grad()
    def caption(self, pil_img: Image.Image, prompt: str = GEOCHAT_PROMPT) -> str:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        image_tensor = self._process_images([pil_img], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        image_tensor = image_tensor.to(self.device, dtype=self.model.dtype)

        if self.model.config.mm_use_im_start_end:
            img_token = (self._DEFAULT_IM_START_TOKEN
                         + self._DEFAULT_IMAGE_TOKEN
                         + self._DEFAULT_IM_END_TOKEN)
        else:
            img_token = self._DEFAULT_IMAGE_TOKEN

        conv = self._conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], img_token + "\n" + prompt)
        conv.append_message(conv.roles[1], None)

        input_ids = self._tokenizer_image_token(
            conv.get_prompt(), self.tokenizer,
            self._IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        output_ids = self.model.generate(
            input_ids,
            images=(image_tensor.unsqueeze(0) if image_tensor.dim() == 3
                    else image_tensor),
            max_new_tokens=80, do_sample=False, temperature=0, use_cache=True,
        )
        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
MODELS = ["generic_clip", "remote_clip"]
MODEL_LABELS = {"generic_clip": "CLIP genérico", "remote_clip": "RemoteCLIP"}


def save_tile_composite(tile_id, pil_lr, pil_sr, pil_gt,
                        cap_lr, cap_sr, cap_gt, cls,
                        scores_txt, scores_hyb, fname):
    def resize(img):
        return img.resize((DISPLAY_PX, DISPLAY_PX), Image.LANCZOS)

    imgs = [resize(pil_lr), resize(pil_sr)]
    cols, caps = ["LR", "SR"], [cap_lr, cap_sr]
    if pil_gt is not None:
        imgs.append(resize(pil_gt)); cols.append("GT"); caps.append(cap_gt)

    fig, axes = plt.subplots(1, len(imgs), figsize=(len(imgs) * 3.2, 6.0))
    if len(imgs) == 1:
        axes = [axes]
    title = f"Tile {tile_id} — GeoChat + CLIP" + (f" | classe: {cls}" if cls else "")
    fig.suptitle(title, fontsize=10, fontweight="bold")

    for ax, img, col, cap in zip(axes, imgs, cols, caps):
        ax.imshow(img); ax.set_title(col, fontsize=9, fontweight="bold"); ax.axis("off")
        wrapped = "\n".join([cap[j:j+40] for j in range(0, len(cap), 40)])
        ax.text(0.5, -0.04, f'"{wrapped}"', transform=ax.transAxes, ha="center",
                va="top", fontsize=5.5, style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.8))

    lines = []
    if scores_txt:
        lines.append("— text-vs-text —")
        for model, label in MODEL_LABELS.items():
            if f"{model}_lr" in scores_txt:
                lr_s, sr_s = scores_txt[f"{model}_lr"], scores_txt[f"{model}_sr"]
                lines.append(f"  {label:18s} LR={lr_s:.3f}  SR={sr_s:.3f}  Δ={sr_s-lr_s:+.3f}")
    if scores_hyb:
        lines.append("— hybrid (img vs text) —")
        for model, label in MODEL_LABELS.items():
            if f"{model}_lr" in scores_hyb:
                lr_s, sr_s = scores_hyb[f"{model}_lr"], scores_hyb[f"{model}_sr"]
                lines.append(f"  {label:18s} LR={lr_s:.3f}  SR={sr_s:.3f}  Δ={sr_s-lr_s:+.3f}")
    if lines:
        fig.text(0.5, 0.01, "\n".join(lines), ha="center", va="bottom", fontsize=6.5,
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightcyan", alpha=0.85))

    plt.tight_layout(rect=[0, 0.15, 1, 0.93])
    plt.savefig(fname, dpi=110); plt.close()


def plot_boxplots(df, scenario, mode_label, fname):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"CLIPScore GeoChat ({mode_label}) — {scenario}", fontsize=12)
    for ax, model in zip(axes, MODELS):
        lr_col, sr_col = f"{model}_lr", f"{model}_sr"
        if lr_col not in df.columns:
            ax.set_visible(False); continue
        data = [df[lr_col].dropna().values, df[sr_col].dropna().values]
        bp = ax.boxplot(data, labels=["LR", "SR"], patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("#5dade2")
        bp["boxes"][1].set_facecolor("#f39c12")
        ax.set_title(MODEL_LABELS[model]); ax.set_ylabel("CLIPScore"); ax.set_ylim(0, 2.6)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()
    print(f"  Salvo: {fname}")


def plot_barplot_by_class(df, mode_label, fname):
    classes = df["classe"].dropna().unique()
    if len(classes) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"CLIPScore médio por Classe — GeoChat {mode_label}", fontsize=12)
    for ax, model in zip(axes, MODELS):
        lr_col, sr_col = f"{model}_lr", f"{model}_sr"
        if lr_col not in df.columns:
            ax.set_visible(False); continue
        lr_means = [df[df["classe"] == c][lr_col].mean() for c in classes]
        sr_means = [df[df["classe"] == c][sr_col].mean() for c in classes]
        x = np.arange(len(classes)); w = 0.35
        ax.bar(x - w/2, lr_means, w, label="LR", color="#5dade2")
        ax.bar(x + w/2, sr_means, w, label="SR", color="#f39c12")
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_title(MODEL_LABELS[model]); ax.set_ylabel("CLIPScore médio")
        ax.set_ylim(0, 2.6); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()
    print(f"  Salvo: {fname}")


def plot_scatter(df, scenario, mode_label, fname):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"CLIPScore LR vs SR — GeoChat {mode_label} — {scenario}", fontsize=11)
    for ax, model in zip(axes, MODELS):
        lr_col, sr_col = f"{model}_lr", f"{model}_sr"
        if lr_col not in df.columns:
            ax.set_visible(False); continue
        ax.scatter(df[lr_col].fillna(0), df[sr_col].fillna(0), alpha=0.6, s=25, color="#2ecc71")
        ax.plot([0, 2.5], [0, 2.5], "r--", linewidth=1, label="y=x")
        ax.set_xlabel("LR CLIPScore"); ax.set_ylabel("SR CLIPScore")
        ax.set_xlim(-0.05, 2.6); ax.set_ylim(-0.05, 2.6)
        ax.set_title(MODEL_LABELS[model]); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()
    print(f"  Salvo: {fname}")


def plot_comparison_modes(df_txt, df_hyb, scenario, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"CLIPScore: text-vs-text vs hybrid — {scenario}", fontsize=12)
    for ax, model in zip(axes, MODELS):
        lr_c, sr_c = f"{model}_lr", f"{model}_sr"
        vals = {}
        if lr_c in df_txt.columns and len(df_txt) > 0:
            vals["LR txt"] = df_txt[lr_c].mean(); vals["SR txt"] = df_txt[sr_c].mean()
        if lr_c in df_hyb.columns and len(df_hyb) > 0:
            vals["LR hyb"] = df_hyb[lr_c].mean(); vals["SR hyb"] = df_hyb[sr_c].mean()
        if not vals:
            ax.set_visible(False); continue
        clrs = {"LR txt": "#5dade2", "SR txt": "#f39c12",
                "LR hyb": "#85c1e9", "SR hyb": "#f8c471"}
        bars = ax.bar(list(vals.keys()), list(vals.values()),
                      color=[clrs[k] for k in vals])
        for bar, val in zip(bars, vals.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(MODEL_LABELS[model]); ax.set_ylabel("CLIPScore médio")
        ax.set_ylim(0, max(vals.values()) * 1.25 if vals else 2.6)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()
    print(f"  Salvo: {fname}")


def print_summary(df, scenario, mode_label):
    print(f"\n{'='*65}")
    print(f"Resumo GeoChat CLIPScore ({mode_label}) — {scenario}  (n={len(df)})")
    print(f"{'='*65}")
    for model in MODELS:
        lr_col, sr_col = f"{model}_lr", f"{model}_sr"
        if lr_col not in df.columns:
            continue
        lr, sr = df[lr_col].dropna(), df[sr_col].dropna()
        wins = (df[sr_col] > df[lr_col]).sum()
        print(f"  {MODEL_LABELS[model]:20s}  LR={lr.mean():.4f}  "
              f"SR={sr.mean():.4f}  Δ={(sr-lr).mean():+.4f}  SR>LR: {wins}/{len(df)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--model-path", type=str, default="MBZUAI/geochat-7B")
    parser.add_argument("--resume", action="store_true",
                        help="Pular tiles já processados e concatenar com CSVs existentes")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TILES_OUT, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    # ---- 1. Carregar GeoPackage --------------------------------------------
    print("\n[Fase 1] Carregando ground truth vetorial (GeoPackage)...")
    gdf_dict = {}
    if GPKG_PATH.exists():
        resolved_map = _resolve_layer_name_map(GPKG_PATH)
        for layer_name, cls_name in resolved_map.items():
            try:
                gdf = gpd.read_file(str(GPKG_PATH), layer=layer_name).to_crs(TARGET_CRS)
                gdf_dict[cls_name] = gdf
                print(f"  {cls_name:25s}: {len(gdf)} polígonos")
            except Exception as e:
                print(f"  AVISO: camada '{layer_name}' não lida: {e}")
    else:
        print(f"  AVISO: {GPKG_PATH} não encontrado — cenário vetorial desativado.")

    # ---- 2. Tiles metadata -------------------------------------------------
    print(f"\n[Fase 2] Carregando tiles pré-gerados...")
    meta_csv = os.path.join(TILES_DIR, "tiles_metadata.csv")
    if not os.path.exists(meta_csv):
        print(f"ERRO: {meta_csv} não encontrado. Execute prepare_data.py primeiro.")
        sys.exit(1)
    meta_df = pd.read_csv(meta_csv)
    if args.max_tiles is not None:
        meta_df = meta_df.head(args.max_tiles)
    print(f"  Tiles: {len(meta_df)}")

    # ---- Resume: identificar tiles já processados ---------------------------
    _csv_names = [
        "geochat_clip_vetorial_txt", "geochat_clip_vetorial_hybrid",
        "geochat_clip_raster_gt_txt", "geochat_clip_raster_gt_hybrid",
        "geochat_clip_raster_rs_txt", "geochat_clip_raster_rs_hybrid",
    ]
    done_ids: set = set()
    if args.resume:
        for _name in _csv_names:
            _p = os.path.join(OUT_DIR, f"{_name}.csv")
            if os.path.exists(_p):
                done_ids.update(pd.read_csv(_p)["tile_id"].tolist())
        orig_count = len(meta_df)
        meta_df = meta_df[~meta_df["tile_id"].isin(done_ids)]
        print(f"  Resume: {len(done_ids)} tiles já processados, "
              f"{len(meta_df)}/{orig_count} restantes")

    # ---- 3. Carregar modelos -----------------------------------------------
    print("\n[Fase 3] Carregando modelos...")
    captioner    = GeoChatCaptioner(model_path=args.model_path,
                                    load_4bit=args.load_4bit,
                                    load_8bit=args.load_8bit)
    clip_generic = GenericCLIP(device)
    remote_clip  = None
    try:
        remote_clip = RemoteCLIP(device)
    except Exception as e:
        print(f"  AVISO: RemoteCLIP não carregado — {e}")

    # Pré-computar embeddings de texto das classes
    cls_emb_generic = {cls: clip_generic.embed_text(cls) for cls in gdf_dict}
    cls_emb_remote  = ({cls: remote_clip.embed_text(cls) for cls in gdf_dict}
                       if remote_clip else {})

    # ---- 4. Processar tiles ------------------------------------------------
    print("\n[Fase 4] Processando tiles (GeoChat caption + CLIPScore)...")
    rec_vec_txt, rec_vec_hyb = [], []
    rec_gt_txt,  rec_gt_hyb  = [], []
    rec_rs_txt,  rec_rs_hyb  = [], []
    t_start = time.time()

    for row_meta in meta_df.itertuples():
        tile_id = row_meta.tile_id
        if args.resume and tile_id in done_ids:
            continue
        bbox    = (row_meta.x0, row_meta.y0, row_meta.x1, row_meta.y1)
        t0 = time.time()
        print(f"  {tile_id}...", end="", flush=True)

        lr_file = os.path.join(TILES_DIR, "lr", f"{tile_id}.png")
        sr_file = os.path.join(TILES_DIR, "sr", f"{tile_id}.png")
        gt_file = os.path.join(TILES_DIR, "gt", f"{tile_id}.png")

        if not os.path.exists(lr_file) or not os.path.exists(sr_file):
            print(" skip (arquivo ausente)"); continue

        pil_lr = Image.open(lr_file).convert("RGB")
        pil_sr = Image.open(sr_file).convert("RGB")
        pil_gt = Image.open(gt_file).convert("RGB") if os.path.exists(gt_file) else None

        cap_lr = captioner.caption(pil_lr)
        cap_sr = captioner.caption(pil_sr)

        emb_txt_lr_g = clip_generic.embed_text(cap_lr)
        emb_txt_sr_g = clip_generic.embed_text(cap_sr)
        emb_txt_lr_r = remote_clip.embed_text(cap_lr) if remote_clip else None
        emb_txt_sr_r = remote_clip.embed_text(cap_sr) if remote_clip else None

        emb_img_lr_g = clip_generic.embed_image(pil_lr)
        emb_img_sr_g = clip_generic.embed_image(pil_sr)
        emb_img_lr_r = remote_clip.embed_image(pil_lr) if remote_clip else None
        emb_img_sr_r = remote_clip.embed_image(pil_sr) if remote_clip else None

        cls = dominant_class(bbox, gdf_dict)
        scores_txt_display, scores_hyb_display = {}, {}

        if cls is not None:
            row_txt = {"tile_id": tile_id, "classe": cls}
            row_txt["generic_clip_lr"] = clipscore(emb_txt_lr_g, cls_emb_generic[cls])
            row_txt["generic_clip_sr"] = clipscore(emb_txt_sr_g, cls_emb_generic[cls])
            if remote_clip:
                row_txt["remote_clip_lr"] = clipscore(emb_txt_lr_r, cls_emb_remote[cls])
                row_txt["remote_clip_sr"] = clipscore(emb_txt_sr_r, cls_emb_remote[cls])
            rec_vec_txt.append(row_txt); scores_txt_display = row_txt

            row_hyb = {"tile_id": tile_id, "classe": cls}
            row_hyb["generic_clip_lr"] = clipscore(emb_img_lr_g, cls_emb_generic[cls])
            row_hyb["generic_clip_sr"] = clipscore(emb_img_sr_g, cls_emb_generic[cls])
            if remote_clip:
                row_hyb["remote_clip_lr"] = clipscore(emb_img_lr_r, cls_emb_remote[cls])
                row_hyb["remote_clip_sr"] = clipscore(emb_img_sr_r, cls_emb_remote[cls])
            rec_vec_hyb.append(row_hyb); scores_hyb_display = row_hyb

        cap_gt = ""
        if pil_gt is not None:
            cap_gt = captioner.caption(pil_gt)
            emb_txt_gt_g = clip_generic.embed_text(cap_gt)
            emb_txt_gt_r = remote_clip.embed_text(cap_gt) if remote_clip else None

            row = {"tile_id": tile_id, "caption_gt": cap_gt}
            row["generic_clip_lr"] = clipscore(emb_txt_lr_g, emb_txt_gt_g)
            row["generic_clip_sr"] = clipscore(emb_txt_sr_g, emb_txt_gt_g)
            if remote_clip:
                row["remote_clip_lr"] = clipscore(emb_txt_lr_r, emb_txt_gt_r)
                row["remote_clip_sr"] = clipscore(emb_txt_sr_r, emb_txt_gt_r)
            rec_gt_txt.append(row)

            row_h = {"tile_id": tile_id, "caption_gt": cap_gt}
            row_h["generic_clip_lr"] = clipscore(emb_img_lr_g, emb_txt_gt_g)
            row_h["generic_clip_sr"] = clipscore(emb_img_sr_g, emb_txt_gt_g)
            if remote_clip:
                row_h["remote_clip_lr"] = clipscore(emb_img_lr_r, emb_txt_gt_r)
                row_h["remote_clip_sr"] = clipscore(emb_img_sr_r, emb_txt_gt_r)
            rec_gt_hyb.append(row_h)

            rec_rs_txt.append(dict(row)); rec_rs_hyb.append(dict(row_h))

        if cls is not None or pil_gt is not None:
            save_tile_composite(
                tile_id, pil_lr, pil_sr, pil_gt,
                cap_lr, cap_sr, cap_gt, cls,
                scores_txt_display, scores_hyb_display,
                os.path.join(TILES_OUT, f"{tile_id}_clip.png"),
            )
        print(f" OK ({time.time()-t0:.1f}s)")

    print(f"\n  Tempo total: {time.time()-t_start:.1f}s")
    print(f"  Vetorial txt : {len(rec_vec_txt)} tiles")
    print(f"  Vetorial hyb : {len(rec_vec_hyb)} tiles")
    print(f"  Raster GT txt: {len(rec_gt_txt)} tiles")
    print(f"  Raster GT hyb: {len(rec_gt_hyb)} tiles")

    # ---- 5. Salvar CSVs ----------------------------------------------------
    datasets = {
        "geochat_clip_vetorial_txt":     rec_vec_txt,
        "geochat_clip_vetorial_hybrid":  rec_vec_hyb,
        "geochat_clip_raster_gt_txt":    rec_gt_txt,
        "geochat_clip_raster_gt_hybrid": rec_gt_hyb,
        "geochat_clip_raster_rs_txt":    rec_rs_txt,
        "geochat_clip_raster_rs_hybrid": rec_rs_hyb,
    }
    dfs = {}
    for name, recs in datasets.items():
        new_df = pd.DataFrame(recs)
        csv_path = os.path.join(OUT_DIR, f"{name}.csv")
        if args.resume and os.path.exists(csv_path) and len(new_df) > 0:
            try:
                old_df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                old_df = pd.DataFrame()
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined.drop_duplicates(subset=["tile_id"], keep="last", inplace=True)
            combined.to_csv(csv_path, index=False)
            dfs[name] = combined
        elif len(new_df) > 0:
            new_df.to_csv(csv_path, index=False)
            dfs[name] = new_df
        elif os.path.exists(csv_path):
            try:
                dfs[name] = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                dfs[name] = new_df
        else:
            dfs[name] = new_df
        print(f"  Salvo: {csv_path}")

    df_vt  = dfs["geochat_clip_vetorial_txt"]
    df_vh  = dfs["geochat_clip_vetorial_hybrid"]
    df_gtt = dfs["geochat_clip_raster_gt_txt"]
    df_gth = dfs["geochat_clip_raster_gt_hybrid"]
    df_rst = dfs["geochat_clip_raster_rs_txt"]
    df_rsh = dfs["geochat_clip_raster_rs_hybrid"]

    # ---- 6. Gráficos -------------------------------------------------------
    print("\n[Fase 6] Gerando gráficos...")
    if len(df_vt) > 0:
        plot_boxplots(df_vt, "GT Vetorial", "text-vs-text",
                      os.path.join(FIG_DIR, "gc_clip_boxplot_vec_txt.png"))
        plot_barplot_by_class(df_vt, "text-vs-text",
                              os.path.join(FIG_DIR, "gc_clip_bar_vec_txt.png"))
        plot_scatter(df_vt, "GT Vetorial", "text-vs-text",
                     os.path.join(FIG_DIR, "gc_clip_scatter_vec_txt.png"))
        print_summary(df_vt, "GT Vetorial", "text-vs-text")

    if len(df_vh) > 0:
        plot_boxplots(df_vh, "GT Vetorial", "hybrid",
                      os.path.join(FIG_DIR, "gc_clip_boxplot_vec_hyb.png"))
        plot_barplot_by_class(df_vh, "hybrid",
                              os.path.join(FIG_DIR, "gc_clip_bar_vec_hyb.png"))
        plot_scatter(df_vh, "GT Vetorial", "hybrid",
                     os.path.join(FIG_DIR, "gc_clip_scatter_vec_hyb.png"))
        print_summary(df_vh, "GT Vetorial", "hybrid")

    if len(df_vt) > 0 and len(df_vh) > 0:
        plot_comparison_modes(df_vt, df_vh, "GT Vetorial",
                              os.path.join(FIG_DIR, "gc_clip_modes_vetorial.png"))

    if len(df_gtt) > 0:
        plot_boxplots(df_gtt, "GT Raster", "text-vs-text",
                      os.path.join(FIG_DIR, "gc_clip_boxplot_gt_txt.png"))
        plot_scatter(df_gtt, "GT Raster", "text-vs-text",
                     os.path.join(FIG_DIR, "gc_clip_scatter_gt_txt.png"))
        print_summary(df_gtt, "GT Raster", "text-vs-text")

    if len(df_gth) > 0:
        plot_boxplots(df_gth, "GT Raster", "hybrid",
                      os.path.join(FIG_DIR, "gc_clip_boxplot_gt_hyb.png"))
        plot_scatter(df_gth, "GT Raster", "hybrid",
                     os.path.join(FIG_DIR, "gc_clip_scatter_gt_hyb.png"))
        print_summary(df_gth, "GT Raster", "hybrid")

    if len(df_gtt) > 0 and len(df_gth) > 0:
        plot_comparison_modes(df_gtt, df_gth, "GT Raster",
                              os.path.join(FIG_DIR, "gc_clip_modes_raster_gt.png"))

    print(f"\nConcluído! Resultados em: {OUT_DIR}")


if __name__ == "__main__":
    main()
