"""
prepare_data.py
===============
Prepara os tiles pareados (LR / SR / GT) para o pipeline de avaliação semântica.

Fluxo:
  1. Reprojeta LR e GT para EPSG:3857
  2. Gera mosaico dos arquivos GT (suporta múltiplos .tif em data/gt/)
  3. Calcula intersecção LR ∩ SR ∩ GT
  4. Gera grid de tiles de TILE_PX_LR × TILE_PX_LR px no espaço LR
  5. Extrai patches LR/SR/GT na resolução nativa de cada imagem
  6. Descarta tiles com >50% pixels NoData
  7. Salva em results/tiles/{lr,sr,gt}/ como PNG

Saídas:
  results/tiles/lr/tile_XXXX.png
  results/tiles/sr/tile_XXXX.png
  results/tiles/gt/tile_XXXX.png
  results/tiles/tiles_metadata.csv

CONFIGURAÇÃO:
  Edite a seção "Configuração" abaixo com os nomes dos seus arquivos.
  LR_PATH: imagem Sentinel-2 de baixa resolução (.jp2 ou .tif)
  SR_PATH: imagem super-resolvida (.tif)
  GT_DIR : diretório com as ortofotos ground truth (.tif) — pode ser um ou vários
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.io import MemoryFile

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuração — EDITAR AQUI com os nomes dos seus arquivos
# ---------------------------------------------------------------------------
BASE    = Path(__file__).parent

# Coloque o nome do arquivo LR dentro de data/lr/
LR_PATH = next((BASE / "data/lr").glob("*.jp2"), None) or \
          next((BASE / "data/lr").glob("*.tif"), None)

# Coloque o nome do arquivo SR dentro de data/sr/
SR_PATH = next((BASE / "data/sr").glob("*.tif"), None)

# Todos os .tif em data/gt/ serão mosaicados automaticamente
GT_DIR  = BASE / "data/gt"
OUT_DIR = BASE / "results/tiles"

GT_FILES    = sorted(GT_DIR.glob("*.tif"))
TARGET_CRS  = "EPSG:3857"
TILE_PX_LR  = 64     # tamanho do tile em pixels no espaço LR
NODATA_THR  = 0.5    # descartar tile se >50% pixels zero

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def reproject_to_memory(src_path, target_crs):
    """Reprojeta um raster para target_crs e retorna um MemoryFile."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update({"crs": target_crs, "transform": transform,
                     "width": width, "height": height, "driver": "GTiff"})
        memfile = MemoryFile()
        with memfile.open(**meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
        return memfile


def mosaic_gt(gt_files, target_crs):
    """Reprojeta e mosaica os arquivos GT; retorna MemoryFile."""
    reprojected = [reproject_to_memory(f, target_crs) for f in gt_files]
    datasets = [mf.open() for mf in reprojected]
    mosaic_arr, mosaic_transform = merge(datasets, method="first")
    meta = datasets[0].meta.copy()
    meta.update({
        "driver": "GTiff",
        "height": mosaic_arr.shape[1],
        "width":  mosaic_arr.shape[2],
        "transform": mosaic_transform,
        "crs": target_crs,
    })
    for ds in datasets:
        ds.close()
    memfile = MemoryFile()
    with memfile.open(**meta) as dst:
        dst.write(mosaic_arr)
    return memfile


def intersect_bounds(*bounds_list):
    """Retorna a bbox de intersecção de uma lista de BoundingBox."""
    left   = max(b.left   for b in bounds_list)
    bottom = max(b.bottom for b in bounds_list)
    right  = min(b.right  for b in bounds_list)
    top    = min(b.top    for b in bounds_list)
    if left >= right or bottom >= top:
        raise ValueError("Sem intersecção entre as imagens.")
    return left, bottom, right, top


def extract_patch(src, left, bottom, right, top):
    """Extrai patch dentro da bbox geográfica; retorna array HWC uint8 RGB."""
    window = from_bounds(left, bottom, right, top, src.transform)
    data = src.read(window=window, boundless=True, fill_value=0)
    data = data[:3]  # garante 3 bandas RGB (descarta alpha se existir)
    data = np.clip(data, 0, 255).astype(np.uint8)
    return np.moveaxis(data, 0, -1)  # (bands, H, W) → (H, W, bands)


def nodata_fraction(patch):
    """Fração de pixels completamente zerados (NoData/borda)."""
    return np.all(patch == 0, axis=-1).mean()


def normalize_contrast(patch):
    """Estica contraste percentílico p2-p98 por banda."""
    out = np.zeros_like(patch, dtype=np.uint8)
    for c in range(patch.shape[2]):
        band = patch[:, :, c].astype(np.float32)
        valid = band[band > 0]
        if valid.size == 0:
            continue
        p2, p98 = np.percentile(valid, (2, 98))
        p98 = max(p98, p2 + 1)
        out[:, :, c] = np.clip((band - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== prepare_data.py ===\n")

    for split in ("lr", "sr", "gt"):
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    # 1. Reprojetar LR e GT para EPSG:3857
    print("Reprojetando LR para EPSG:3857...")
    lr_mem = reproject_to_memory(LR_PATH, TARGET_CRS)

    print(f"Mosaicando e reprojetando {len(GT_FILES)} imagens GT...")
    gt_mem = mosaic_gt(GT_FILES, TARGET_CRS)

    lr_ds = lr_mem.open()
    gt_ds = gt_mem.open()
    sr_ds = rasterio.open(SR_PATH)

    print(f"\n  LR res: {lr_ds.res[0]:.2f} m/px  bounds: {lr_ds.bounds}")
    print(f"  SR res: {sr_ds.res[0]:.2f} m/px  bounds: {sr_ds.bounds}")
    print(f"  GT res: {gt_ds.res[0]:.2f} m/px  bounds: {gt_ds.bounds}")

    # 2. Intersecção
    left, bottom, right, top = intersect_bounds(lr_ds.bounds, sr_ds.bounds, gt_ds.bounds)
    print(f"\nIntersecção: {right-left:.0f} m × {top-bottom:.0f} m")
    print(f"  left={left:.1f}  bottom={bottom:.1f}  right={right:.1f}  top={top:.1f}")

    # 3. Grid de tiles baseado na resolução LR
    tile_m = TILE_PX_LR * lr_ds.res[0]
    xs = np.arange(left,   right - tile_m + 1, tile_m)
    ys = np.arange(bottom, top   - tile_m + 1, tile_m)
    print(f"\nTile: {TILE_PX_LR} px LR = {tile_m:.0f} m")
    print(f"Grid: {len(xs)} × {len(ys)} = {len(xs)*len(ys)} tiles potenciais\n")

    records = []
    kept = 0

    for y0 in ys:
        for x0 in xs:
            x1, y1 = x0 + tile_m, y0 + tile_m

            patch_lr = extract_patch(lr_ds, x0, y0, x1, y1)
            patch_gt = extract_patch(gt_ds, x0, y0, x1, y1)

            if nodata_fraction(patch_lr) > NODATA_THR:
                continue
            if nodata_fraction(patch_gt) > NODATA_THR:
                continue

            patch_sr = extract_patch(sr_ds, x0, y0, x1, y1)

            patch_lr = normalize_contrast(patch_lr)
            patch_sr = normalize_contrast(patch_sr)
            patch_gt = normalize_contrast(patch_gt)

            tile_id = f"tile_{kept:04d}"
            Image.fromarray(patch_lr).save(OUT_DIR / "lr" / f"{tile_id}.png")
            Image.fromarray(patch_sr).save(OUT_DIR / "sr" / f"{tile_id}.png")
            Image.fromarray(patch_gt).save(OUT_DIR / "gt" / f"{tile_id}.png")

            records.append({
                "tile_id": tile_id,
                "x0": round(x0, 2), "y0": round(y0, 2),
                "x1": round(x1, 2), "y1": round(y1, 2),
                "lr_shape": f"{patch_lr.shape[0]}x{patch_lr.shape[1]}",
                "sr_shape": f"{patch_sr.shape[0]}x{patch_sr.shape[1]}",
                "gt_shape": f"{patch_gt.shape[0]}x{patch_gt.shape[1]}",
            })
            kept += 1
            print(f"  {kept} tiles salvos...", end="\r")

    lr_ds.close(); sr_ds.close(); gt_ds.close()
    lr_mem.close(); gt_mem.close()

    pd.DataFrame(records).to_csv(OUT_DIR / "tiles_metadata.csv", index=False)

    print(f"\n\nTiles válidos : {kept} / {len(xs)*len(ys)}")
    print(f"Metadados     : {OUT_DIR / 'tiles_metadata.csv'}")
    print("Concluído.")


if __name__ == "__main__":
    main()
