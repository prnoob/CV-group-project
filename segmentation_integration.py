# -*- coding: utf-8 -*-
"""
segmentation_integration.py

Glue code to integrate plate.py (plate segmentation) and food.py (food-vs-plate separation)
into your training/inference pipeline (e.g., mini-cnn.ipynb).
"""
import os, cv2, numpy as np, pandas as pd
from pathlib import Path
import importlib.util

def _import_from_path(py_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_FOOD = _import_from_path(str(Path("food.py")), "food_mod")
_PLATE = _import_from_path(str(Path("plate.py")), "plate_mod")

def _ensure_uint8_mask(mask_bool):
    if mask_bool is None:
        return None
    if mask_bool.dtype != np.uint8:
        mask_bool = mask_bool.astype(np.uint8)
    return (mask_bool > 0).astype(np.uint8) * 255

def segment_row(row, root_dir=".", plate_method="recommended", save_dir=None):
    rgb_path   = Path(root_dir) / str(row['rgb_path']).replace('\\','/')
    depth_path = Path(root_dir) / str(row['depth_raw_path']).replace('\\','/')

    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read RGB: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth: {depth_path}")

    if plate_method == "recommended":
        plate_mask_bool = _PLATE.segment_recommended(depth)
    elif plate_method == "center":
        plate_mask_bool = _PLATE.segment_with_center_bias(depth, threshold_percentile=78)
    elif plate_method == "improved":
        plate_mask_bool = _PLATE.segment_depth_based_improved(depth, threshold_percentile=78)
        plate_mask_bool = _PLATE.clean_mask_aggressive(plate_mask_bool, min_area=15000)
    else:
        plate_mask_bool = _FOOD.segment_plate_region(depth, threshold_percentile=70)

    plate_mask = _ensure_uint8_mask(plate_mask_bool)
    food_mask  = _ensure_uint8_mask(_FOOD.separate_food_from_plate(rgb, depth, plate_mask_bool.astype(bool)))
    background_mask = cv2.bitwise_not(plate_mask)

    overlay = rgb.copy().astype(np.float32)
    if plate_mask is not None:
        m = (plate_mask>0)[...,None].astype(np.float32)
        overlay = overlay*(1-0.45*m) + np.array([0,255,255], np.float32)[None,None,:]*0.45*m
    if food_mask is not None:
        m = (food_mask>0)[...,None].astype(np.float32)
        overlay = overlay*(1-0.45*m) + np.array([255,0,255], np.float32)[None,None,:]*0.45*m
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    out = {
        "rgb": rgb,
        "depth": depth,
        "plate_mask": plate_mask,
        "food_mask": food_mask,
        "background_mask": background_mask,
        "overlay": overlay,
    }

    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        stem = str(row.get("dish_id", Path(rgb_path).stem))
        cv2.imwrite(str(save_dir/f"{stem}_plate.png"), plate_mask)
        cv2.imwrite(str(save_dir/f"{stem}_food.png"), food_mask)
        cv2.imwrite(str(save_dir/f"{stem}_background.png"), background_mask)
        cv2.imwrite(str(save_dir/f"{stem}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return out

def build_masks_for_csv(csv_path, root_dir=".", out_dir="seg_outputs", max_rows=None,
                        plate_method="recommended"):
    df = pd.read_csv(csv_path)
    n = len(df) if max_rows is None else min(max_rows, len(df))
    records = []
    for i in range(n):
        row = df.iloc[i].to_dict()
        dish_id = row.get("dish_id", f"row_{i}")
        save_dir = Path(out_dir) / str(dish_id)
        try:
            out = segment_row(row, root_dir=root_dir, plate_method=plate_method, save_dir=save_dir)
            rec = {
                "dish_id": dish_id,
                "rgb": str(Path(root_dir) / str(row["rgb_path"])),
                "depth": str(Path(root_dir) / str(row["depth_raw_path"])),
                "plate_px": int((out["plate_mask"]>0).sum()),
                "food_px": int((out["food_mask"]>0).sum()),
                "plate_cov": float((out["plate_mask"]>0).mean()),
                "food_of_plate": float(((out["food_mask"]>0) & (out["plate_mask"]>0)).sum()) / max(1, (out["plate_mask"]>0).sum()),
                "saved_dir": str(save_dir.resolve())
            }
            records.append(rec)
        except Exception as e:
            records.append({"dish_id": dish_id, "error": str(e)})
    return pd.DataFrame(records)

def make_supervised_npz(csv_path, root_dir=".", out_dir="seg_outputs", classes=("background","plate","food"),
                        max_rows=None, resize_to=None):
    df = pd.read_csv(csv_path)
    n = len(df) if max_rows is None else min(max_rows, len(df))

    X_list, y_list, ids = [], [], []
    for i in range(n):
        row = df.iloc[i].to_dict()
        out = segment_row(row, root_dir=root_dir, plate_method="recommended", save_dir=None)
        rgb = out["rgb"]
        plate = (out["plate_mask"]>0).astype(np.uint8)
        food  = (out["food_mask"]>0).astype(np.uint8)

        y = np.zeros(plate.shape, np.uint8)
        y[plate==1] = 1
        y[food==1]  = 2

        if resize_to is not None:
            H, W = resize_to
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
            y   = cv2.resize(y,   (W, H), interpolation=cv2.INTER_NEAREST)

        X_list.append(rgb); y_list.append(y); ids.append(row.get("dish_id", f"row_{i}"))

    X = np.stack(X_list, 0); y = np.stack(y_list, 0)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "dataset.npz"
    np.savez_compressed(str(out_npz), X=X, y=y, ids=np.array(ids), classes=np.array(classes))
    return str(out_npz)
