#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ResNet 回归（从零训练，无预训练）- 稳定优化完整版
- 仅读取已有的 train_index.csv / test_index.csv（不在此脚本内构造索引）
- 三种输入（互斥可消融）：RGB+depth_color(6ch) / RGB+depth_raw(4ch) / RGB+grad3(6ch)
- plate 权重：用 depth_raw 估计盘面覆盖率，样本损失按 (1 + gamma * coverage) 加权
- 标签 z-score 仅用于损失（指标仍用原尺度），加速稳定收敛
- 头部预热 + Warmup + Cosine，Huber/MSE，梯度裁剪，SWA（推理默认用 SWA）
- 原生 6 通道 conv1（--native6）或 1x1 stem
- 预测含水平翻转 TTA
- 自动保存：metrics CSV、训练曲线、best 模型、test/train 预测 CSV（train 带 GT 评估）
"""

import argparse, math, os, random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode as IM

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# ---------------------------
# Utilities
# ---------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_path(images_root: Optional[Path], p: str) -> Path:
    """将 CSV 中的相对路径拼到 images_root；若已是绝对路径则直接返回。"""
    p_clean = str(p).strip().strip('"').strip("'").replace("\\", "/")
    path = Path(p_clean)
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
                 split: str,
                 img_size: int = 224,
                 use_depth_color: bool = False,
                 use_depth_raw: bool = False,
                 use_depth_grad3: bool = False,
                 plate_weight_gamma: float = 1.0):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.split = split
        self.use_depth_color = use_depth_color
        self.use_depth_raw = use_depth_raw
        self.use_depth_grad3 = use_depth_grad3
        self.plate_weight_gamma = float(plate_weight_gamma)
        self.img_size = img_size

        # RGB 只做颜色类增强；几何增强我们手动对齐到 depth
        self.color_aug = None
        if split == "train":
            self.color_aug = T.Compose([
                T.TrivialAugmentWide(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            ])
        self.rand_erasing = T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))

    def __len__(self): return len(self.df)

    # ---- depth helpers ----
    def _load_depth_color_tensor(self, path_opt: Optional[str], H: int, W: int) -> torch.Tensor:
        if not self.use_depth_color or not path_opt or (isinstance(path_opt, float) and np.isnan(path_opt)):
            return torch.zeros(3, H, W)
        p = resolve_path(self.images_root, str(path_opt))
        if not p.exists(): return torch.zeros(3, H, W)
        img = Image.open(p).convert("RGB").resize((W, H))
        t = TF.to_tensor(img)
        return (t - 0.5) / 0.25

    def _load_depth_raw_01_tensor(self, path_opt: Optional[str], H: int, W: int) -> torch.Tensor:
        """深度 raw -> [1,H,W] in [0,1]"""
        if not path_opt or (isinstance(path_opt, float) and np.isnan(path_opt)):
            return torch.zeros(1, H, W)
        p = resolve_path(self.images_root, str(path_opt))
        if not p.exists(): return torch.zeros(1, H, W)
        try:
            img = Image.open(p)
            if img.mode in ("I;16","I"):
                arr = np.array(img, dtype=np.uint16).astype(np.float32)
                if arr.max() > 0: arr = arr / 65535.0
            else:
                arr = np.array(img.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
            arr = np.clip(arr, 0.0, 1.0)
            arr_img = Image.fromarray((arr*255).astype(np.uint8)).resize((W, H))
            return TF.to_tensor(arr_img).to(torch.float32)
        except:
            return torch.zeros(1, H, W)

    def _depth_grad3(self, d01: torch.Tensor) -> torch.Tensor:
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        dx = F.conv2d(d01.unsqueeze(0), kx, padding=1).squeeze(0)
        dy = F.conv2d(d01.unsqueeze(0), ky, padding=1).squeeze(0)
        def _norm01(x):
            mu, sd = x.mean(), x.std().clamp_min(1e-6)
            z = (x - mu) / sd
            z = z.clamp(-3, 3)
            return (z + 3) / 6
        D = d01; Dx = _norm01(dx); Dy = _norm01(dy)
        feat = torch.cat([D, Dx, Dy], dim=0)  # [3,H,W]
        return (feat - 0.5) / 0.25

    def _plate_mask_from_depth(self, d01: torch.Tensor, k: float = 2.5) -> torch.Tensor:
        """median+MAD 估计盘面/桌面，返回 [1,H,W] 0/1 mask"""
        d = d01[0]; H, W = d.shape
        h0,h1 = int(0.3*H), int(0.7*H); w0,w1 = int(0.3*W), int(0.7*W)
        center = d[h0:h1, w0:w1]
        m = center.median()
        mad = (center - m).abs().median().clamp_min(1e-6)
        mask = ((d - m).abs() <= (k*mad)).float().unsqueeze(0)
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        return (mask > 0.5).float()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # RGB 路径
        if "rgb_path" in row and not pd.isna(row["rgb_path"]): rgb_col = "rgb_path"
        elif "image_path" in row and not pd.isna(row["image_path"]): rgb_col = "image_path"
        else: raise KeyError("Row missing 'rgb_path' or 'image_path'.")

        rgb_path = resolve_path(self.images_root, str(row[rgb_col]))
        if not rgb_path.exists(): raise FileNotFoundError(f"RGB not found: {rgb_path}")
        img_rgb = Image.open(rgb_path).convert("RGB")

        # 统一几何增强（train 随机裁剪+翻转；val/test 中心裁剪）
        if self.split == "train":
            i,j,h,w = T.RandomResizedCrop.get_params(img_rgb, scale=(0.8,1.0), ratio=(3/4,4/3))
            do_flip = random.random() < 0.5
            rgb_aug = TF.resized_crop(img_rgb, i,j,h,w, (self.img_size,self.img_size), interpolation=IM.BILINEAR)
            if self.color_aug is not None: rgb_aug = self.color_aug(rgb_aug)
            if do_flip: rgb_aug = TF.hflip(rgb_aug)
        else:
            rgb_aug = TF.center_crop(TF.resize(img_rgb, (self.img_size,self.img_size), interpolation=IM.BILINEAR),
                                     [self.img_size,self.img_size])

        x_rgb = TF.to_tensor(rgb_aug)
        x_rgb = TF.normalize(x_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        # random erasing 仅作用于 RGB
        if self.split == "train":
            x_rgb = self.rand_erasing(x_rgb)
        C,H,W = x_rgb.shape
        channels = [x_rgb]

        # depth_color
        if self.use_depth_color:
            t_dcol = self._load_depth_color_tensor(row.get("depth_color_path", None), H, W)
            channels.append(t_dcol)

        # depth_raw（含 grad3）
        d01 = torch.zeros(1, H, W)
        if ("depth_raw_path" in row) and not pd.isna(row["depth_raw_path"]):
            d01 = self._load_depth_raw_01_tensor(row["depth_raw_path"], H, W)

        if self.use_depth_raw:
            channels.append((d01 - 0.5) / 0.25)
        if self.use_depth_grad3:
            channels.append(self._depth_grad3(d01))

        x = torch.cat(channels, dim=0)  # [C_total,H,W]

        # 样本权重（盘子覆盖率）
        sample_weight = 1.0
        if self.plate_weight_gamma > 0:
            m_plate = self._plate_mask_from_depth(d01)
            coverage = float(m_plate.mean().item())
            sample_weight = 1.0 + self.plate_weight_gamma * coverage
        sample_weight = torch.tensor([sample_weight], dtype=torch.float32)

        if "label" in self.df.columns and self.split in ("train","val"):
            y = torch.tensor([float(row["label"])], dtype=torch.float32)
            sid = str(row.get("ID", idx))
            return x, y, sid, sample_weight
        else:
            sid = str(row.get("ID", row.get("ID", idx)))
            return x, sid

# ---------------------------
# Model (no pretraining)
# ---------------------------
class ResNetRegressor(nn.Module):
    def __init__(self, in_ch=3, backbone="resnet18",
                 native6=False, mlp_hidden: Optional[List[int]]=None,
                 mlp_dropout: float=0.0, mlp_act: str="relu"):
        super().__init__()
        # 不使用预训练
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=None)
            feat_dim = self.backbone.fc.in_features
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=None)
            feat_dim = self.backbone.fc.in_features
        else:
            raise ValueError("Unsupported backbone (use resnet18 or resnet50).")

        # 输入通道处理
        if native6 and in_ch != 3:
            old = self.backbone.conv1
            new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                # 从零初始化；可用 Kaiming
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
            self.backbone.conv1 = new
            self.stem = None
        else:
            self.stem = (nn.Sequential(
                nn.Conv2d(in_ch, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            ) if in_ch != 3 else None)

        self.backbone.fc = nn.Identity()

        # MLP 头
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[mlp_act.lower()]
        hidden = [] if (mlp_hidden is None or len(mlp_hidden)==0) else mlp_hidden
        layers = []
        in_dim = feat_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), act(), nn.Dropout(mlp_dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        if self.stem is not None: x = self.stem(x)
        feat = self.backbone(x)
        return self.head(feat)

    def init_output_bias(self, label_mean: float):
        """最后线性层：bias=label均值，weight=0，帮助稳启动。"""
        last = None
        for m in reversed(self.head):
            if isinstance(m, nn.Linear):
                last = m; break
        if last is not None:
            nn.init.constant_(last.bias, float(label_mean))
            nn.init.zeros_(last.weight)

# ---------------------------
# Loss / Train / Val / Predict
# ---------------------------
def weighted_huber(pred, target, weight, delta: float):
    diff = pred - target
    abs_diff = diff.abs()
    loss = torch.where(abs_diff <= delta, 0.5 * diff * diff,
                       delta * (abs_diff - 0.5 * delta))
    return (loss * weight).sum() / (weight.sum() + 1e-8)

def weighted_mse(pred, target, weight):
    se = (pred - target) ** 2
    return (se * weight).sum() / (weight.sum() + 1e-8)

def compute_unweighted_metrics(pred, target):
    se = (pred - target) ** 2
    ae = (pred - target).abs()
    return se.sum().item(), ae.sum().item()

def set_requires_grad(module: nn.Module, flag: bool):
    if module is None: return
    for p in module.parameters(): p.requires_grad = flag

def build_optimizer(model, base_lr, weight_decay, head_lr_mult=1.0, only_head=False):
    head_params = list(model.head.parameters())
    stem_params = list(model.stem.parameters()) if model.stem is not None else []
    backbone_params = [p for n,p in model.named_parameters()
                       if ("head" not in n and "stem" not in n)]

    if only_head:
        params = [{"params": head_params, "lr": base_lr*head_lr_mult, "weight_decay": weight_decay}]
    else:
        params = [
            {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": stem_params,     "lr": base_lr, "weight_decay": weight_decay},
            {"params": head_params,     "lr": base_lr*head_lr_mult, "weight_decay": weight_decay},
        ]
    return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

def build_scheduler(optimizer, epochs, warmup_epochs):
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
    warmup = LambdaLR(optimizer, lr_lambda=lambda ep: min(1.0, (ep+1)/max(1,warmup_epochs)))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    if warmup_epochs > 0:
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return cosine

def train_one_epoch(model, loader, optimizer, device, args, y_mean, y_std, grad_clip):
    model.train()
    mse_sum=0.0; mae_sum=0.0; n=0
    for batch in loader:
        x,y,_,w = batch
        x,y,w = x.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        # loss 用 z 空间；指标走原空间
        if args.label_norm:
            p = model(x)
            p_loss = (p - y_mean) / y_std
            y_loss = (y - y_mean) / y_std
        else:
            p = model(x); p_loss = p; y_loss = y

        if args.loss == "huber":
            loss = weighted_huber(p_loss, y_loss, w, args.huber_delta)
        else:
            loss = weighted_mse(p_loss, y_loss, w)

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        se_sum, ae_sum = compute_unweighted_metrics(p, y)
        mse_sum += se_sum; mae_sum += ae_sum; n += x.size(0)

    mse = mse_sum / max(1,n)
    rmse = math.sqrt(max(0.0, mse))
    mae = mae_sum / max(1,n)
    return mse, rmse, mae

@torch.no_grad()
def validate(model, loader, device, args, y_mean, y_std):
    model.eval()
    mse_sum=0.0; mae_sum=0.0; n=0
    for batch in loader:
        x,y,_,_ = batch
        x,y = x.to(device), y.to(device)
        p = model(x)
        se_sum, ae_sum = compute_unweighted_metrics(p, y)
        mse_sum += se_sum; mae_sum += ae_sum; n += x.size(0)
    mse = mse_sum / max(1,n)
    rmse = math.sqrt(max(0.0, mse))
    mae = mae_sum / max(1,n)
    return mse, rmse, mae

@torch.no_grad()
def predict_tta_hflip(model, loader, device) -> List[Tuple[str, float]]:
    """推理端 TTA：原图 + 水平翻转取均值"""
    model.eval()
    out=[]
    for batch in loader:
        if len(batch)==2: x,ids = batch
        else:             x,_,ids,_ = batch
        x = x.to(device)
        y0 = model(x)
        y1 = model(torch.flip(x, dims=[-1]))
        y  = (y0 + y1) * 0.5
        out += list(zip(list(map(str, ids)), y.squeeze(1).cpu().numpy().astype(float).tolist()))
    return out

# ---------------------------
# Plot & Save
# ---------------------------
def save_metrics_and_plot(hist, out_dir: Path):
    df = pd.DataFrame(hist)
    csv_path = out_dir / "metrics_trainval.csv"
    df.to_csv(csv_path, index=False)
    # 图
    fig = plt.figure(figsize=(7,5))
    plt.plot(df["epoch"], df["tr_mse"], label="train MSE")
    plt.plot(df["epoch"], df["va_mse"], label="val MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    fig_path = out_dir / "train_val_mse.png"
    plt.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"[OK] metrics CSV -> {csv_path}")
    print(f"[OK] figure      -> {fig_path}")

def write_predictions_csv(pairs: List[Tuple[str,float]], path: Path, with_label_df: Optional[pd.DataFrame]=None):
    if with_label_df is None:
        pd.DataFrame({"ID":[i for i,_ in pairs], "Value":[v for _,v in pairs]}).to_csv(path, index=False)
        print(f"[OK] wrote predictions -> {path} (rows={len(pairs)})")
    else:
        df = pd.DataFrame({"ID":[i for i,_ in pairs], "pred":[v for _,v in pairs]})
        df = df.merge(with_label_df[["ID","label"]], on="ID", how="left")
        df["se"] = (df["pred"] - df["label"])**2
        df["ae"] = (df["pred"] - df["label"]).abs()
        mse = float(df["se"].mean()); mae = float(df["ae"].mean()); rmse = math.sqrt(mse)
        df.to_csv(path, index=False)
        print(f"[TRAIN PRED] rows={len(df)}  MSE={mse:.4f} RMSE={rmse:.4f} MAE={mae:.4f}")
        print(f"[OK] wrote train predictions with GT -> {path}")

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv",  required=False, default=None)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=0)
    # 输入模态
    ap.add_argument("--use_depth_color", action="store_true")
    ap.add_argument("--use_depth_raw", action="store_true")
    ap.add_argument("--use_depth_grad3", action="store_true")
    ap.add_argument("--plate_weight_gamma", type=float, default=1.0)
    # 模型
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--native6", action="store_true", help="直接把conv1改为in_ch输入；否则用1x1 stem")
    ap.add_argument("--mlp_hidden", type=str, default="256,64")
    ap.add_argument("--mlp_dropout", type=float, default=0.2)
    ap.add_argument("--mlp_act", type=str, default="gelu", choices=["relu","gelu","silu"])
    # 训练
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--head_warmup_epochs", type=int, default=2)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--loss", type=str, default="huber", choices=["mse","huber"])
    ap.add_argument("--huber_delta", type=float, default=50.0, help="若 --label_norm 开启，建议 1.0~2.0")
    ap.add_argument("--label_norm", action="store_true", help="对label做z-score，仅用于loss")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # SWA
    ap.add_argument("--swa", action="store_true")
    ap.add_argument("--swa_start", type=float, default=0.6, help="从训练进度比例开始做SWA")
    # 输出
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--pred_csv", type=str, default="predictions_test.csv")
    ap.add_argument("--train_pred_csv", type=str, default="predictions_train.csv")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    images_root = Path(args.images_root).resolve()

    # 读取 CSV（只使用你已有的索引）
    train_df = pd.read_csv(args.train_csv)
    if not {"ID","rgb_path","label"}.issubset(train_df.columns):
        raise ValueError("train_csv 需包含列：ID, rgb_path, label（可选：depth_color_path, depth_raw_path）")
    # 随机划分 val
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_total = len(train_df); n_val = max(1, int(n_total * args.val_split))
    df_val = train_df.iloc[:n_val].reset_index(drop=True)
    df_trn = train_df.iloc[n_val:].reset_index(drop=True)

    # 标签均值方差（用于 z 空间）
    y_mean = float(train_df["label"].mean())
    y_std_raw = float(pd.to_numeric(train_df["label"], errors="coerce").std(skipna=True))
    y_std = max(y_std_raw, 1e-8)


    # 数据集/加载器
    ds_trn = IndexDishDataset(df_trn, images_root, split="train",
                              img_size=args.img_size,
                              use_depth_color=args.use_depth_color,
                              use_depth_raw=args.use_depth_raw,
                              use_depth_grad3=args.use_depth_grad3,
                              plate_weight_gamma=args.plate_weight_gamma)
    ds_val = IndexDishDataset(df_val, images_root, split="val",
                              img_size=args.img_size,
                              use_depth_color=args.use_depth_color,
                              use_depth_raw=args.use_depth_raw,
                              use_depth_grad3=args.use_depth_grad3,
                              plate_weight_gamma=args.plate_weight_gamma)

    # 探针通道
    probe_x = ds_trn[0][0]
    in_ch = int(probe_x.shape[0])
    print(f"Train size: {len(ds_trn)} | Val size: {len(ds_val)} | in_ch: {in_ch} | device: {device}")

    pin_memory = torch.cuda.is_available()  # MPS/CPU 不开
    dl_trn = DataLoader(ds_trn, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=pin_memory)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=pin_memory)

    # 模型（无预训练）
    hidden = [int(x) for x in args.mlp_hidden.split(",") if x.strip()]
    model = ResNetRegressor(in_ch=in_ch, backbone=args.backbone,
                            native6=args.native6,
                            mlp_hidden=hidden, mlp_dropout=args.mlp_dropout, mlp_act=args.mlp_act).to(device)
    model.init_output_bias(y_mean)

    # 优化器 & 调度器
    opt = build_optimizer(model, args.lr, args.weight_decay, args.head_lr_mult, only_head=(args.head_warmup_epochs>0))
    sched = build_scheduler(opt, args.epochs, args.warmup_epochs)

    # SWA
    if args.swa:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_model = AveragedModel(model)
        swa_start_epoch = int(args.epochs * args.swa_start)
        swa_scheduler = SWALR(opt, anneal_strategy="cos", anneal_epochs=5, swa_lr=args.lr*0.5)
    else:
        swa_model = None

    best_rmse = float("inf")
    best_path = out_dir / "best_model.pth"
    hist = []

    for ep in range(1, args.epochs+1):
        # 头部预热
        only_head = (ep <= args.head_warmup_epochs)
        set_requires_grad(model.backbone, not only_head)
        set_requires_grad(model.stem,     not only_head)
        set_requires_grad(model.head,      True)

        # 切换优化器（只在过渡边界切一次最安全）
        if ep == args.head_warmup_epochs + 1:
            opt = build_optimizer(model, args.lr, args.weight_decay, args.head_lr_mult, only_head=False)
            sched = build_scheduler(opt, args.epochs, args.warmup_epochs)

        tr_mse, tr_rmse, tr_mae = train_one_epoch(model, dl_trn, opt, device, args, y_mean, y_std, args.grad_clip)
        va_mse, va_rmse, va_mae = validate(model, dl_val, device, args, y_mean, y_std)

        # scheduler / SWA
        if args.swa and ep > swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            sched.step()

        print(f"Epoch {ep:03d}/{args.epochs} | "
              f"TR MSE: {tr_mse:.4f} RMSE: {tr_rmse:.4f} MAE: {tr_mae:.4f} | "
              f"VA MSE: {va_mse:.4f} RMSE: {va_rmse:.4f} MAE: {va_mae:.4f}")

        hist.append({"epoch":ep, "tr_mse":tr_mse, "tr_rmse":tr_rmse, "tr_mae":tr_mae,
                              "va_mse":va_mse, "va_rmse":va_rmse, "va_mae":va_mae})

        if va_rmse < best_rmse:
            best_rmse = va_rmse
            torch.save({"model_state": model.state_dict(),
                        "in_ch": in_ch,
                        "backbone": args.backbone,
                        "img_size": args.img_size}, best_path)
            print(f"  ↳ Saved best: {best_path} (val RMSE {best_rmse:.4f})")

    save_metrics_and_plot(hist, out_dir)

    # 选择推理模型：SWA 优先，否则加载 best
    active_model = model
    if args.swa:
        from torch.optim.swa_utils import update_bn
        # 用训练集 BN 更新
        update_bn(dl_trn, swa_model, device=device)
        active_model = swa_model
        torch.save({"model_state": active_model.module.state_dict() if hasattr(active_model, "module") else active_model.state_dict(),
                    "in_ch": in_ch, "backbone": args.backbone, "img_size": args.img_size},
                   out_dir / "swa_model.pth")
        print(f"[OK] saved SWA model -> {out_dir/'swa_model.pth'}")
    else:
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            active_model = model

    # ---- 训练集全量预测（带 GT 评估）----
    dl_trn_full = DataLoader(IndexDishDataset(train_df, images_root, split="val",
                                              img_size=args.img_size,
                                              use_depth_color=args.use_depth_color,
                                              use_depth_raw=args.use_depth_raw,
                                              use_depth_grad3=args.use_depth_grad3,
                                              plate_weight_gamma=args.plate_weight_gamma),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)
    train_pairs = predict_tta_hflip(active_model, dl_trn_full, device)
    # 合并 GT 并写 CSV（含 MSE/MAE/RMSE）
    write_predictions_csv(train_pairs, out_dir / args.train_pred_csv, with_label_df=train_df)

    # ---- 测试集预测 ----
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv)
        if "rgb_path" not in test_df.columns and "image_path" not in test_df.columns:
            raise ValueError("test_csv 需包含 'rgb_path' 或 'image_path' 列")
        ds_tst = IndexDishDataset(test_df, images_root, split="test",
                                  img_size=args.img_size,
                                  use_depth_color=args.use_depth_color,
                                  use_depth_raw=args.use_depth_raw,
                                  use_depth_grad3=args.use_depth_grad3,
                                  plate_weight_gamma=args.plate_weight_gamma)
        dl_tst = DataLoader(ds_tst, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)
        test_pairs = predict_tta_hflip(active_model, dl_tst, device)
        write_predictions_csv(test_pairs, out_dir / args.pred_csv)

if __name__ == "__main__":
    main()
