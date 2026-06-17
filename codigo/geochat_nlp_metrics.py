import os
import tempfile
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
geochat_nlp_metrics.py
======================
Compara métricas NLP entre legendas geradas pelo GeoChat (MBZUAI/geochat-7B)
para imagens Sentinel-2 de baixa resolução (LR) vs super-resolvidas (SR).

Três cenários de avaliação:
  A) GT vetorial          : referência = nome da classe de cobertura do solo (GeoPackage)
  B) GT raster            : referência = legenda gerada para a imagem de alta res (gt.tif)
  C) GT raster reamostrado: referência = legenda gerada para GT na resolução da SR

Fonte vetorial: data/vetores_cobertura_solo.gpkg (múltiplas camadas, uma por classe)

Saídas:
  results/geochat_metrics_vetorial.csv
  results/geochat_metrics_raster_gt.csv
  results/geochat_metrics_raster_gt_resampled.csv
  results/figures/geochat_*.png
  results/tiles_geochat/tile_XXXX.png

Uso:
  python geochat_nlp_metrics.py --model-path models/geochat-7B
  python geochat_nlp_metrics.py --model-path models/geochat-7B --max-tiles 5
  python geochat_nlp_metrics.py --model-path models/geochat-7B --load-4bit
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
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.io import MemoryFile
import geopandas as gpd
from shapely.geometry import box
from PIL import Image
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — auto-detecção a partir da pasta do script
# ---------------------------------------------------------------------------
BASE    = Path(__file__).resolve().parent

LR_PATH = str(
    next((BASE / "data/lr").glob("*.jp2"), None) or
    next((BASE / "data/lr").glob("*.tif"), None)
)
SR_PATH  = str(next((BASE / "data/sr").glob("*.tif"), None))
GT_FILES = sorted((BASE / "data/gt").glob("*.tif"))

# GeoPackage — camadas mapeadas para nomes de classe legíveis
GPKG_PATH = BASE / "data" / "vetores_cobertura_solo.gpkg"

# Mapeamento canônico: nome-da-camada-no-gpkg → classe legível.
# O GDAL/QGIS às vezes preserva ".shp" no nome da camada ao importar shapefiles;
# o script tenta ambas as variantes (com e sem sufixo) automaticamente.
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
    """Retorna um mapa {nome_real_no_gpkg: classe} resolvendo variantes com/sem .shp."""
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
        if layer_key in available:                                   # 1. exato
            resolved[layer_key] = cls_name
        elif layer_key + ".shp" in available:                        # 2. + .shp
            resolved[layer_key + ".shp"] = cls_name
        elif layer_key.removesuffix(".shp") in available:            # 3. - .shp
            resolved[layer_key.removesuffix(".shp")] = cls_name
        else:
            avail_bases = {a: a.removesuffix(".shp").lower() for a in available}
            exact_ci = next((a for a, b in avail_bases.items() if b == base), None)
            if exact_ci:                                             # 4. case-insensitive
                resolved[exact_ci] = cls_name
            else:
                # 5. sufixo — trata prefixos duplicados (ex: VEG_Veg_Cultivada_A)
                suffix_m = next((a for a, b in avail_bases.items()
                                 if b.endswith("_" + base) or b.endswith(base)), None)
                if suffix_m:
                    print(f"  NOTA: '{layer_key}' → '{suffix_m}' (match por sufixo)")
                    resolved[suffix_m] = cls_name
                else:
                    print(f"  AVISO: camada '{layer_key}' não encontrada no GeoPackage")
    return resolved

OUT_DIR      = str(BASE / "results")
FIG_DIR      = os.path.join(OUT_DIR, "figures")
TILES_DIR    = os.path.join(OUT_DIR, "tiles")
TILES_OUT    = os.path.join(OUT_DIR, "tiles_geochat")
TARGET_CRS   = "EPSG:3857"

TILE_PX     = 64
NODATA_FRAC = 0.5
DISPLAY_PX  = 256

GEOCHAT_PROMPT = (
    "Describe this remote sensing image in one sentence. "
    "Focus on the land cover type visible."
)

_nltk_dir = os.environ.get("NLTK_DATA")
if _nltk_dir and _nltk_dir not in nltk.data.path:
    nltk.data.path.insert(0, _nltk_dir)
for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True, raise_on_error=False)


# ---------------------------------------------------------------------------
# Reprojeção
# ---------------------------------------------------------------------------
# Usa arquivo temporário no disco com BIGTIFF=YES para suportar GT > 4 GB
def _tmpfile_reproject(src_path: str, target_res_m=None):
    from rasterio.transform import from_bounds as _from_bounds
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds
        )
        if target_res_m is not None:
            left   = transform.c
            top    = transform.f
            right  = left  + width  * abs(transform.a)
            bottom = top   - height * abs(transform.e)
            width  = max(1, int(round((right - left)  / target_res_m)))
            height = max(1, int(round((top   - bottom) / target_res_m)))
            transform = _from_bounds(left, bottom, right, top, width, height)

        profile = src.profile.copy()
        profile.update(
            crs=TARGET_CRS, transform=transform,
            width=width, height=height,
            driver="GTiff", compress="lzw", BIGTIFF="YES",
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp_path = tmp.name
        tmp.close()
        n_bands = src.count
        size_gb = os.path.getsize(src_path) / 1e9
        print(f"    reprojetando {n_bands} banda(s) ({size_gb:.1f} GB)...", flush=True)
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for i in range(1, n_bands + 1):
                print(f"    banda {i}/{n_bands}...", end="\r", flush=True)
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=TARGET_CRS,
                    resampling=Resampling.bilinear,
                )
        print(f"    {n_bands} banda(s) reprojetadas.          ", flush=True)
    return tmp_path


def mosaic_gt(gt_files, target_crs, target_res_m=None):
    from rasterio.merge import merge as rasterio_merge
    from rasterio.transform import from_bounds as _from_bounds

    # Atalho: 1 arquivo + sem reamostragem → abre direto, sem merge em RAM
    if len(gt_files) == 1 and target_res_m is None:
        p = _tmpfile_reproject(str(gt_files[0]), target_res_m=None)
        ds_out = rasterio.open(p)
        return p, ds_out

    tmp_paths = []
    datasets  = []
    for f in gt_files:
        p = _tmpfile_reproject(str(f), target_res_m=None)
        tmp_paths.append(p)
        datasets.append(rasterio.open(p))

    mosaic_arr, mosaic_transform = rasterio_merge(datasets, method="first")
    mosaic_arr = mosaic_arr[:3]  # garante RGB (descarta alpha se existir)

    meta = datasets[0].meta.copy()
    meta.update({
        "driver": "GTiff", "count": 3, "BIGTIFF": "YES",
        "height": mosaic_arr.shape[1],
        "width":  mosaic_arr.shape[2],
        "transform": mosaic_transform,
        "crs": target_crs, "compress": "lzw",
    })

    if target_res_m is not None:
        left   = mosaic_transform.c
        top    = mosaic_transform.f
        right  = left + mosaic_arr.shape[2] * abs(mosaic_transform.a)
        bottom = top  - mosaic_arr.shape[1] * abs(mosaic_transform.e)
        w = max(1, int(round((right - left)  / target_res_m)))
        h = max(1, int(round((top   - bottom) / target_res_m)))
        new_transform = _from_bounds(left, bottom, right, top, w, h)
        new_arr = np.zeros((3, h, w), dtype=mosaic_arr.dtype)
        from rasterio.warp import reproject as _reproject
        for band_i in range(3):
            _reproject(
                source=mosaic_arr[band_i],
                destination=new_arr[band_i],
                src_transform=mosaic_transform, src_crs=target_crs,
                dst_transform=new_transform,   dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
        mosaic_arr = new_arr
        mosaic_transform = new_transform
        meta.update({"height": h, "width": w, "transform": new_transform})

    for ds in datasets:
        ds.close()
    for p in tmp_paths:
        try:
            os.unlink(p)
        except OSError:
            pass

    # Gravar mosaico final em arquivo temporário (pode ser > 4 GB)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp_out_path = tmp_out.name
    tmp_out.close()
    with rasterio.open(tmp_out_path, "w", **meta) as dst:
        dst.write(mosaic_arr)

    ds_out = rasterio.open(tmp_out_path)
    return tmp_out_path, ds_out


# ---------------------------------------------------------------------------
# Extração de patch — sempre retorna RGB
# ---------------------------------------------------------------------------
def extract_patch(dataset, bbox):
    try:
        window = from_bounds(*bbox, transform=dataset.transform)
        window = window.intersection(
            rasterio.windows.Window(0, 0, dataset.width, dataset.height)
        )
        if window.width < 2 or window.height < 2:
            return None
        bands = min(dataset.count, 3)
        data = dataset.read(
            list(range(1, bands + 1)),
            window=window,
            out_shape=(bands, max(1, int(window.height)), max(1, int(window.width))),
            resampling=Resampling.bilinear,
        )
        if bands < 3:
            data = np.concatenate([data] * (3 // bands + 1), axis=0)[:3]
        if np.all(data == 0, axis=0).mean() > NODATA_FRAC:
            return None
        arr = data.astype(np.float32)
        p2, p98 = (np.percentile(arr[arr > 0], (2, 98))
                   if (arr > 0).any() else (0, 1))
        arr = np.clip((arr - p2) / max(p98 - p2, 1e-6) * 255, 0, 255).astype(np.uint8)
        return arr.transpose(1, 2, 0)
    except Exception:
        return None


def to_pil(arr):
    return Image.fromarray(arr[:, :, :3].astype(np.uint8))


# ---------------------------------------------------------------------------
# Classe dominante via GeoPackage
# ---------------------------------------------------------------------------
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
# GeoChat captioning
# ---------------------------------------------------------------------------
class GeoChatCaptioner:
    def __init__(self, model_path: str = "MBZUAI/geochat-7B",
                 load_4bit: bool = False, load_8bit: bool = False):

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
        self._conv_templates          = conv_templates
        self._tokenizer_image_token   = tokenizer_image_token
        self._process_images          = process_images
        self._IMAGE_TOKEN_INDEX       = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN     = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN  = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN    = DEFAULT_IM_END_TOKEN
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

        full_prompt = img_token + "\n" + prompt
        conv = self._conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = self._tokenizer_image_token(
            prompt_text, self.tokenizer,
            self._IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor,
            max_new_tokens=80,
            do_sample=False, temperature=0, use_cache=True,
        )
        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()


# ---------------------------------------------------------------------------
# NLP metrics
# ---------------------------------------------------------------------------
smoother = SmoothingFunction().method1

def bleu1(hypothesis: str, reference: str) -> float:
    hyp_tok = hypothesis.lower().split()
    ref_tok = reference.lower().split()
    if not hyp_tok or not ref_tok:
        return 0.0
    return sentence_bleu([ref_tok], hyp_tok, weights=(1, 0, 0, 0),
                         smoothing_function=smoother)

def rouge1(hypothesis: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    return scorer.score(reference, hypothesis)["rouge1"].fmeasure

def tfidf_cosine(hypothesis: str, reference: str) -> float:
    try:
        vec = TfidfVectorizer().fit_transform([hypothesis, reference])
        return float(cosine_similarity(vec[0], vec[1])[0, 0])
    except Exception:
        return 0.0

def compute_metrics(hyp: str, ref: str) -> dict:
    return {"bleu1": bleu1(hyp, ref), "rouge1": rouge1(hyp, ref), "tfidf": tfidf_cosine(hyp, ref)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
METRICS = ["bleu1", "rouge1", "tfidf"]
METRIC_LABELS = {"bleu1": "BLEU-1", "rouge1": "ROUGE-1 F1", "tfidf": "TF-IDF Cosine"}


def save_tile_composite(tile_id, pil_lr, pil_sr, pil_gt,
                        cap_lr, cap_sr, cap_gt,
                        cls, m_lr_vec, m_sr_vec,
                        m_lr_rgt, m_sr_rgt, fname):
    def resize(img):
        return img.resize((DISPLAY_PX, DISPLAY_PX), Image.LANCZOS)

    imgs = [resize(pil_lr), resize(pil_sr)]
    cols, caps = ["LR", "SR"], [cap_lr, cap_sr]
    if pil_gt is not None:
        imgs.append(resize(pil_gt)); cols.append("GT raster"); caps.append(cap_gt)

    fig, axes = plt.subplots(1, len(imgs), figsize=(len(imgs) * 3.2, 5.5))
    if len(imgs) == 1:
        axes = [axes]
    title = f"Tile {tile_id} — GeoChat" + (f" | classe: {cls}" if cls else "")
    fig.suptitle(title, fontsize=10, fontweight="bold")

    for ax, img, col, cap in zip(axes, imgs, cols, caps):
        ax.imshow(img); ax.set_title(col, fontsize=9, fontweight="bold"); ax.axis("off")
        wrapped = "\n".join([cap[j:j+40] for j in range(0, len(cap), 40)])
        ax.text(0.5, -0.04, f'"{wrapped}"', transform=ax.transAxes, ha="center",
                va="top", fontsize=6, style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.8))

    lines = []
    if m_lr_vec and cls:
        lines.append(f"vs classe  →  LR: BLEU={m_lr_vec['bleu1']:.3f} "
                     f"ROUGE={m_lr_vec['rouge1']:.3f} TF-IDF={m_lr_vec['tfidf']:.3f}")
        lines.append(f"             SR: BLEU={m_sr_vec['bleu1']:.3f} "
                     f"ROUGE={m_sr_vec['rouge1']:.3f} TF-IDF={m_sr_vec['tfidf']:.3f}")
    if m_lr_rgt and pil_gt is not None:
        lines.append(f"vs GT rast →  LR: BLEU={m_lr_rgt['bleu1']:.3f} "
                     f"ROUGE={m_lr_rgt['rouge1']:.3f} TF-IDF={m_lr_rgt['tfidf']:.3f}")
        lines.append(f"             SR: BLEU={m_sr_rgt['bleu1']:.3f} "
                     f"ROUGE={m_sr_rgt['rouge1']:.3f} TF-IDF={m_sr_rgt['tfidf']:.3f}")
    if lines:
        fig.text(0.5, 0.01, "\n".join(lines), ha="center", va="bottom", fontsize=7,
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightcyan", alpha=0.85))

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(fname, dpi=110); plt.close()


def plot_boxplots(df, scenario, fname):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Distribuição Métricas NLP (GeoChat) — {scenario}", fontsize=13)
    for ax, m in zip(axes, METRICS):
        data = [df[f"{m}_lr"].dropna().values, df[f"{m}_sr"].dropna().values]
        bp = ax.boxplot(data, labels=["LR", "SR"], patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("#5dade2")
        bp["boxes"][1].set_facecolor("#f39c12")
        ax.set_title(METRIC_LABELS[m]); ax.set_ylabel("Score"); ax.set_ylim(-0.05, 1.05)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()
    print(f"  Salvo: {fname}")


def plot_barplot_by_class(df, fname):
    classes = df["classe"].dropna().unique()
    if len(classes) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(5 * len(classes) // 2 + 6, 5))
    fig.suptitle("Média Métricas por Classe (GeoChat) — GT Vetorial", fontsize=13)
    for ax, m in zip(axes, METRICS):
        lr_means = [df[df["classe"] == c][f"{m}_lr"].mean() for c in classes]
        sr_means = [df[df["classe"] == c][f"{m}_sr"].mean() for c in classes]
        x = np.arange(len(classes)); w = 0.35
        ax.bar(x - w/2, lr_means, w, label="LR", color="#5dade2")
        ax.bar(x + w/2, sr_means, w, label="SR", color="#f39c12")
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS[m]); ax.set_ylabel("Score médio")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()
    print(f"  Salvo: {fname}")


def plot_scatter(df, scenario, fname):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"LR vs SR por tile (GeoChat) — {scenario}", fontsize=13)
    for ax, m in zip(axes, METRICS):
        lr, sr = df[f"{m}_lr"].fillna(0), df[f"{m}_sr"].fillna(0)
        ax.scatter(lr, sr, alpha=0.5, s=20)
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="y=x")
        ax.set_xlabel(f"LR {METRIC_LABELS[m]}"); ax.set_ylabel(f"SR {METRIC_LABELS[m]}")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_title(METRIC_LABELS[m]); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()
    print(f"  Salvo: {fname}")


def print_summary(df, scenario):
    print(f"\n{'='*65}")
    print(f"Resumo GeoChat — {scenario}  (n={len(df)} tiles)")
    print(f"{'='*65}")
    for m in METRICS:
        lr, sr = df[f"{m}_lr"].dropna(), df[f"{m}_sr"].dropna()
        print(f"  {METRIC_LABELS[m]:20s}  LR={lr.mean():.4f}±{lr.std():.4f}  "
              f"SR={sr.mean():.4f}±{sr.std():.4f}  Δ={(sr-lr).mean():+.4f}")


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

    # ---- 1. Reprojetar imagens ----------------------------------------------
    print("\n[Fase 1] Reprojetando imagens para EPSG:3857...")
    tmp_lr = _tmpfile_reproject(LR_PATH)
    ds_lr  = rasterio.open(tmp_lr)
    with rasterio.open(SR_PATH) as _p:
        _same = rasterio.crs.CRS.from_user_input(TARGET_CRS) == _p.crs
    if _same:
        print(f"  SR já em {TARGET_CRS}, abrindo sem reprojetar...")
        tmp_sr, ds_sr = None, rasterio.open(SR_PATH)
    else:
        tmp_sr = _tmpfile_reproject(SR_PATH)
        ds_sr  = rasterio.open(tmp_sr)
    gt_size_gb = sum(os.path.getsize(f) for f in GT_FILES) / 1e9
    print(f"  Mosaicando {len(GT_FILES)} arquivo(s) GT ({gt_size_gb:.1f} GB)...")
    tmp_gt, ds_gt = mosaic_gt(GT_FILES, TARGET_CRS)
    sr_res_m = abs(ds_sr.transform.a)
    print(f"  Resolução SR detectada: {sr_res_m:.4f} m/px")
    tmp_gt_rs, ds_gt_rs = mosaic_gt(GT_FILES, TARGET_CRS, target_res_m=sr_res_m)

    # ---- 2. Carregar GeoPackage ---------------------------------------------
    print("\n[Fase 2] Carregando ground truth vetorial (GeoPackage)...")
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

    # ---- 3. Carregar tiles pré-gerados --------------------------------------
    print(f"\n[Fase 3] Carregando tiles pré-gerados de {TILES_DIR}...")
    meta_csv = os.path.join(TILES_DIR, "tiles_metadata.csv")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(
            f"tiles_metadata.csv não encontrado em {TILES_DIR}. "
            "Execute prepare_data.py primeiro."
        )
    meta_df = pd.read_csv(meta_csv)
    print(f"  {len(meta_df)} tiles disponíveis")
    if args.max_tiles is not None:
        meta_df = meta_df.head(args.max_tiles)
        print(f"  Limitado a {len(meta_df)} tiles (--max-tiles)")

    # ---- Resume: identificar tiles já processados ---------------------------
    _csv_vec = os.path.join(OUT_DIR, "geochat_metrics_vetorial.csv")
    _csv_gt  = os.path.join(OUT_DIR, "geochat_metrics_raster_gt.csv")
    _csv_rs  = os.path.join(OUT_DIR, "geochat_metrics_raster_gt_resampled.csv")
    done_ids: set = set()
    if args.resume:
        for _csv in [_csv_vec, _csv_gt, _csv_rs]:
            if os.path.exists(_csv):
                try:
                    done_ids.update(pd.read_csv(_csv)["tile_id"].tolist())
                except pd.errors.EmptyDataError:
                    pass
        orig_count = len(meta_df)
        meta_df = meta_df[~meta_df["tile_id"].isin(done_ids)]
        print(f"  Resume: {len(done_ids)} tiles já processados, "
              f"{len(meta_df)}/{orig_count} restantes")

    # ---- 4. Carregar GeoChat ------------------------------------------------
    print("\n[Fase 4] Carregando modelo GeoChat... (pode levar 10-15 min)", flush=True)
    _t_model = time.time()
    captioner = GeoChatCaptioner(
        model_path=args.model_path,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
    )
    print(f"  Modelo carregado em {(time.time()-_t_model)/60:.1f} min", flush=True)

    # ---- 5. Processar tiles -------------------------------------------------
    print("\n[Fase 5] Processando tiles e gerando captions com GeoChat...")
    records_vec, records_gt, records_gt_rs = [], [], []
    t_start = time.time()
    _tile_times = []

    for row in meta_df.itertuples():
        idx = row.tile_id
        bbox = (row.x0, row.y0, row.x1, row.y1)
        t_tile = time.time()
        n_done = len(_tile_times)
        n_total = len(meta_df)
        if n_done > 0:
            eta_s = (sum(_tile_times) / n_done) * (n_total - n_done)
            eta_str = f"ETA ~{eta_s/60:.0f} min" if eta_s >= 60 else f"ETA ~{eta_s:.0f}s"
        else:
            eta_str = "ETA calculando..."
        print(f"  [{n_done+1}/{n_total}] tile {idx} | {eta_str} ...",
              end="", flush=True)

        lr_file = os.path.join(TILES_DIR, "lr", f"{idx}.png")
        sr_file = os.path.join(TILES_DIR, "sr", f"{idx}.png")
        gt_file = os.path.join(TILES_DIR, "gt", f"{idx}.png")

        if not os.path.exists(lr_file) or not os.path.exists(sr_file):
            print(f" skip (arquivo ausente)")
            continue

        pil_lr = Image.open(lr_file).convert("RGB")
        pil_sr = Image.open(sr_file).convert("RGB")
        pil_gt = Image.open(gt_file).convert("RGB") if os.path.exists(gt_file) else None

        cap_lr = captioner.caption(pil_lr)
        cap_sr = captioner.caption(pil_sr)

        # Cenário A: GT vetorial
        cls = dominant_class(bbox, gdf_dict)
        m_lr_vec = m_sr_vec = None
        if cls is not None:
            m_lr_vec = compute_metrics(cap_lr, cls)
            m_sr_vec = compute_metrics(cap_sr, cls)
            records_vec.append({
                "tile_id": idx, "classe": cls,
                "caption_lr": cap_lr, "caption_sr": cap_sr,
                "bleu1_lr": m_lr_vec["bleu1"],  "bleu1_sr":  m_sr_vec["bleu1"],
                "rouge1_lr": m_lr_vec["rouge1"], "rouge1_sr": m_sr_vec["rouge1"],
                "tfidf_lr": m_lr_vec["tfidf"],  "tfidf_sr":  m_sr_vec["tfidf"],
            })

        # Cenário B: GT raster
        cap_gt = ""
        m_lr_rgt = m_sr_rgt = None
        if pil_gt is not None:
            cap_gt = captioner.caption(pil_gt)
            m_lr_rgt = compute_metrics(cap_lr, cap_gt)
            m_sr_rgt = compute_metrics(cap_sr, cap_gt)
            records_gt.append({
                "tile_id": idx,
                "caption_lr": cap_lr, "caption_sr": cap_sr, "caption_gt": cap_gt,
                "bleu1_lr": m_lr_rgt["bleu1"],  "bleu1_sr":  m_sr_rgt["bleu1"],
                "rouge1_lr": m_lr_rgt["rouge1"], "rouge1_sr": m_sr_rgt["rouge1"],
                "tfidf_lr": m_lr_rgt["tfidf"],  "tfidf_sr":  m_sr_rgt["tfidf"],
            })

        # Cenário C: GT reamostrado (mesma tile GT)
        if pil_gt is not None and m_lr_rgt is not None:
            records_gt_rs.append({
                "tile_id": idx,
                "caption_lr": cap_lr, "caption_sr": cap_sr, "caption_gt_rs": cap_gt,
                "bleu1_lr": m_lr_rgt["bleu1"],  "bleu1_sr":  m_sr_rgt["bleu1"],
                "rouge1_lr": m_lr_rgt["rouge1"], "rouge1_sr": m_sr_rgt["rouge1"],
                "tfidf_lr": m_lr_rgt["tfidf"],  "tfidf_sr":  m_sr_rgt["tfidf"],
            })

        if cls is not None or pil_gt is not None:
            save_tile_composite(
                idx, pil_lr, pil_sr, pil_gt,
                cap_lr, cap_sr, cap_gt,
                cls, m_lr_vec, m_sr_vec, m_lr_rgt, m_sr_rgt,
                os.path.join(TILES_OUT, f"{idx}_geochat.png"),
            )

        _tile_times.append(time.time() - t_tile)
        print(f" OK ({_tile_times[-1]:.1f}s) cap_lr=\"{cap_lr[:50]}...\"", flush=True)

    print(f"\n  Tempo total: {time.time()-t_start:.1f}s")
    print(f"  Tiles válidos (GT vetorial)  : {len(records_vec)}")
    print(f"  Tiles válidos (GT raster)    : {len(records_gt)}")

    # ---- 6. Salvar CSVs -----------------------------------------------------
    csv_vec = _csv_vec
    csv_gt  = _csv_gt
    csv_rs  = _csv_rs

    if args.resume:
        for csv_path, records in [(csv_vec, records_vec),
                                   (csv_gt,  records_gt),
                                   (csv_rs,  records_gt_rs)]:
            new_df = pd.DataFrame(records)
            if os.path.exists(csv_path) and len(new_df) > 0:
                try:
                    old_df = pd.read_csv(csv_path)
                except pd.errors.EmptyDataError:
                    old_df = pd.DataFrame()
                combined = pd.concat([old_df, new_df], ignore_index=True)
                combined.drop_duplicates(subset=["tile_id"], keep="last", inplace=True)
                combined.to_csv(csv_path, index=False)
            elif len(new_df) > 0:
                new_df.to_csv(csv_path, index=False)
    else:
        pd.DataFrame(records_vec).to_csv(csv_vec, index=False)
        pd.DataFrame(records_gt).to_csv(csv_gt, index=False)
        pd.DataFrame(records_gt_rs).to_csv(csv_rs, index=False)

    print(f"\n  Salvo: {csv_vec}")
    print(f"  Salvo: {csv_gt}")
    print(f"  Salvo: {csv_rs}")

    def _safe_read(path):
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    df_vec   = _safe_read(csv_vec)
    df_gt    = _safe_read(csv_gt)
    df_gt_rs = _safe_read(csv_rs)

    # ---- 7. Gráficos --------------------------------------------------------
    print("\n[Fase 6] Gerando gráficos...")
    if len(df_vec) > 0:
        plot_boxplots(df_vec, "GT Vetorial",
                      os.path.join(FIG_DIR, "geochat_boxplot_vetorial.png"))
        plot_barplot_by_class(df_vec,
                              os.path.join(FIG_DIR, "geochat_barplot_por_classe.png"))
        plot_scatter(df_vec, "GT Vetorial",
                     os.path.join(FIG_DIR, "geochat_scatter_vetorial.png"))
        print_summary(df_vec, "GT Vetorial")

    if len(df_gt) > 0:
        plot_boxplots(df_gt, "GT Raster",
                      os.path.join(FIG_DIR, "geochat_boxplot_raster_gt.png"))
        plot_scatter(df_gt, "GT Raster",
                     os.path.join(FIG_DIR, "geochat_scatter_raster_gt.png"))
        print_summary(df_gt, "GT Raster")

    if len(df_gt_rs) > 0:
        plot_boxplots(df_gt_rs, f"GT Raster reamostrado ({sr_res_m:.2f} m/px)",
                      os.path.join(FIG_DIR, "geochat_boxplot_raster_gt_resampled.png"))
        plot_scatter(df_gt_rs, f"GT Raster reamostrado ({sr_res_m:.2f} m/px)",
                     os.path.join(FIG_DIR, "geochat_scatter_raster_gt_resampled.png"))
        print_summary(df_gt_rs, f"GT Raster reamostrado ({sr_res_m:.2f} m/px)")

    ds_lr.close()
    ds_sr.close()
    ds_gt.close()
    ds_gt_rs.close()
    for _p in [tmp_lr, tmp_sr, tmp_gt, tmp_gt_rs]:
        if _p and os.path.exists(_p):
            try:
                os.unlink(_p)
            except OSError:
                pass

    print(f"\nConcluído! Resultados em: {OUT_DIR}")


if __name__ == "__main__":
    main()
