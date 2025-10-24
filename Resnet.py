import argparse
import math
import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import models

# ---------------------------
# Utilities
# ---------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_path(images_root: Optional[Path], p: str) -> Path:
    """Return absolute path. If p is absolute, return as-is; else join with images_root (or current dir)."""
    path = Path(p)
    if path.is_absolute():
        return path
    base = images_root if images_root is not None else Path(".")
    return (base / path).resolve()

# ---------------------------
# Dataset
# ---------------------------
class IndexDishDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 images_root: Optional[Path],
                 split: str,  # "train" or "val" or "test"
                 img_size: int = 224,
                 use_depth_color: bool = False,
                 use_depth_raw: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.split = split
        self.use_depth_color = use_depth_color
        self.use_depth_raw = use_depth_raw

        # Transforms
        if split == "train":
            self.transform_rgb = T.Compose([
                T.Resize(int(img_size * 1.15)),
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform_rgb = T.Compose([
                T.Resize(int(img_size * 1.15)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def _load_rgb_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform_rgb(img)

    def _load_depth_color_tensor(self, path: Optional[str]) -> Optional[torch.Tensor]:
        if not self.use_depth_color or not path or (isinstance(path, float) and np.isnan(path)):
            return None
        p = resolve_path(self.images_root, str(path))
        if not p.exists():
            return None
        # read as RGB and then normalize ~ similar to RGB branch
        img = Image.open(p).convert("RGB").resize((self.img_size, self.img_size))
        t = T.ToTensor()(img)  # [3,H,W], already in [0,1]
        # simple normalization for depth_color (tend to be pseudo-color); use 0.5/0.25 heuristic
        mean = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
        std  = torch.tensor([0.25, 0.25, 0.25])[:, None, None]
        return (t - mean) / std

    def _load_depth_raw_tensor(self, path: Optional[str]) -> Optional[torch.Tensor]:
        if not self.use_depth_raw or not path or (isinstance(path, float) and np.isnan(path)):
            return None
        p = resolve_path(self.images_root, str(path))
        if not p.exists():
            return None
        # Load as single-channel (grayscale); scale to [0,1] by dividing max 65535 if 16-bit
        try:
            img = Image.open(p)
            # Convert to 16-bit or 8-bit grayscale then normalize to [0,1]
            if img.mode in ("I;16", "I"):
                arr = np.array(img, dtype=np.uint16).astype(np.float32)
                if arr.max() > 0:
                    arr = arr / 65535.0
            else:
                arr = np.array(img.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
            # Resize to (img_size, img_size)
            from PIL import Image as PILImage
            arr_img = PILImage.fromarray((arr * 255.0).clip(0,255).astype(np.uint8)).resize((self.img_size, self.img_size))
            t = T.ToTensor()(arr_img)  # [1,H,W] in [0,1]
            # normalize with mean/std of 0.5/0.25
            mean = torch.tensor([0.5])[:, None, None]
            std  = torch.tensor([0.25])[:, None, None]
            return (t - mean) / std
        except Exception:
            return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # path columns
        if "rgb_path" in row and not pd.isna(row["rgb_path"]):
            rgb_col = "rgb_path"
        elif "image_path" in row and not pd.isna(row["image_path"]):
            rgb_col = "image_path"
        else:
            raise KeyError("Row missing rgb/image path column (expected 'rgb_path' or 'image_path').")

        rgb_path = resolve_path(self.images_root, str(row[rgb_col]))
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")

        x_rgb = self._load_rgb_tensor(rgb_path)
        channels = [x_rgb]

        # optional depth channels
        if "depth_color_path" in row:
            dcol = self._load_depth_color_tensor(row["depth_color_path"])
            if dcol is not None:
                channels.append(dcol)
        if "depth_raw_path" in row:
            draw = self._load_depth_raw_tensor(row["depth_raw_path"])
            if draw is not None:
                channels.append(draw)

        # concatenate if we have extra channels
        if len(channels) == 1:
            x = channels[0]
        else:
            # If extra channels are present, just concat along channel dimension
            # (backbone expects 3-ch; we will handle with a small stem to remap to 3-ch later)
            x = torch.cat(channels, dim=0)  # [C, H, W]

        # target / id
        if "label" in self.df.columns and self.split in ("train", "val"):
            y = torch.tensor([float(row["label"])], dtype=torch.float32)
            sample_id = str(row.get("dish_id", idx))
            return x, y, sample_id
        else:
            # test
            sample_id = str(row.get("ID", row.get("dish_id", idx)))
            return x, sample_id

# ---------------------------
# Model (supports variable input channels by a 1x1 conv stem)
# ---------------------------
class ResNetRegressor(nn.Module):
    def __init__(self, in_ch: int = 3, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feat_dim = self.backbone.fc.in_features
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feat_dim = self.backbone.fc.in_features
        else:
            raise ValueError("Unsupported backbone.")
        # If in_ch != 3, add a stem to 3-ch then feed into backbone
        if in_ch != 3:
            self.stem = nn.Conv2d(in_ch, 3, kernel_size=1, bias=False)
        else:
            self.stem = None
        self.backbone.fc = nn.Linear(feat_dim, 1)

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        return self.backbone(x)

# ---------------------------
# Train / Val / Predict
# ---------------------------
def rmse_from_mse(mse: float) -> float:
    return math.sqrt(max(0.0, mse))

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    for batch in loader:
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        p = model(x)
        loss = F.mse_loss(p, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            bs = x.size(0)
            mse_sum += F.mse_loss(p, y, reduction="sum").item()
            mae_sum += F.l1_loss(p, y, reduction="sum").item()
            n += bs
    mse = mse_sum / max(1, n)
    mae = mae_sum / max(1, n)
    return mse, rmse_from_mse(mse), mae

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    for batch in loader:
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        p = model(x)
        bs = x.size(0)
        mse_sum += F.mse_loss(p, y, reduction="sum").item()
        mae_sum += F.l1_loss(p, y, reduction="sum").item()
        n += bs
    mse = mse_sum / max(1, n)
    mae = mae_sum / max(1, n)
    return mse, rmse_from_mse(mse), mae

@torch.no_grad()
def predict(model, loader, device) -> List[Tuple[str, float]]:
    model.eval()
    out: List[Tuple[str, float]] = []
    for batch in loader:
        if len(batch) == 2:
            x, sample_ids = batch
        else:
            x, _, sample_ids = batch
        x = x.to(device)
        y = model(x).cpu().numpy().reshape(-1)
        out.extend(list(zip(list(sample_ids), [float(v) for v in y])))
    return out

def resolve_path(images_root: Optional[Path], p: str) -> Path:
    """Normalize Windows-style separators on POSIX and prepend images_root if needed."""
    p_clean = str(p).strip().strip('"').strip("'").replace("\\", "/")
    path = Path(p_clean)
    if path.is_absolute():
        return path
    base = images_root if images_root is not None else Path(".")
    return (base / path).resolve()

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train_index.csv")
    parser.add_argument("--test_csv",  type=str, default=None, help="Path to test_index.csv (optional for inference)")
    parser.add_argument("--images_root", type=str, default=None, help="Optional root to prepend to relative paths")
    parser.add_argument("--use_depth_color", action="store_true", help="Stack depth_color as extra 3 channels")
    parser.add_argument("--use_depth_raw", action="store_true", help="Stack depth_raw as extra 1 channel")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--pred_csv", type=str, default="predictions.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    images_root = Path(args.images_root).resolve() if args.images_root else None

    # Load train index
    train_df = pd.read_csv(args.train_csv)
    needed_cols = {"dish_id", "label"}
    if not needed_cols.issubset(set(train_df.columns)):
        raise ValueError(f"train_index.csv must contain: {needed_cols} plus path columns (rgb_path or image_path). Found: {train_df.columns.tolist()}")

    # Shuffle and split
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_total = len(train_df)
    n_val = max(1, int(n_total * args.val_split))
    df_val = train_df.iloc[:n_val].reset_index(drop=True)
    df_trn = train_df.iloc[n_val:].reset_index(drop=True)

    # Derive input channels based on flags
    in_ch = 3
    if args.use_depth_color:
        in_ch += 3
    if args.use_depth_raw:
        in_ch += 1

    # Datasets / loaders
    ds_trn = IndexDishDataset(df_trn, images_root, split="train",
                              img_size=args.img_size,
                              use_depth_color=args.use_depth_color,
                              use_depth_raw=args.use_depth_raw)
    ds_val = IndexDishDataset(df_val, images_root, split="val",
                              img_size=args.img_size,
                              use_depth_color=args.use_depth_color,
                              use_depth_raw=args.use_depth_raw)
    dl_trn = DataLoader(ds_trn, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Model
    model = ResNetRegressor(in_ch=in_ch, backbone=args.backbone, pretrained=(not args.no_pretrained)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_rmse = float("inf")
    best_path = out_dir / "best_model.pth"

    print(f"Train size: {len(ds_trn)} | Val size: {len(ds_val)} | in_ch: {in_ch}")
    for ep in range(1, args.epochs + 1):
        tr_mse, tr_rmse, tr_mae = train_one_epoch(model, dl_trn, opt, device)
        va_mse, va_rmse, va_mae = validate(model, dl_val, device)
        sch.step()
        print(f"Epoch {ep:03d}/{args.epochs} | "
              f"TR MSE: {tr_mse:.4f} RMSE: {tr_rmse:.4f} MAE: {tr_mae:.4f} | "
              f"VA MSE: {va_mse:.4f} RMSE: {va_rmse:.4f} MAE: {va_mae:.4f}")
        if va_rmse < best_rmse:
            best_rmse = va_rmse
            torch.save({
                "model_state": model.state_dict(),
                "in_ch": in_ch,
                "backbone": args.backbone,
                "img_size": args.img_size
            }, best_path)
            print(f"  ↳ Saved best: {best_path} (val RMSE {best_rmse:.4f})")

    # Inference on test_index.csv if provided
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv)
        if "image_path" not in test_df.columns and "rgb_path" not in test_df.columns:
            raise ValueError("test_index.csv must include 'image_path' or 'rgb_path' column.")
        ds_tst = IndexDishDataset(test_df, images_root, split="test",
                                  img_size=args.img_size,
                                  use_depth_color=args.use_depth_color,
                                  use_depth_raw=args.use_depth_raw)
        dl_tst = DataLoader(ds_tst, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        # Load best weights
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            model.to(device)
        preds = predict(model, dl_tst, device)
        sub = pd.DataFrame({"ID": [i for i, _ in preds],
                            "Value": [v for _, v in preds]})
        out_csv = out_dir / args.pred_csv
        sub.to_csv(out_csv, index=False)
        print(f"Wrote predictions to: {out_csv}")

if __name__ == "__main__":
    main()
