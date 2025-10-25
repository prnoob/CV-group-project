#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ResNet 回归 • 稳启动 + 抗异常 + 更稳优化（完整可运行版）
- 三种输入（互斥可消融）：RGB+depth_color(6ch) / RGB+depth_raw(4ch) / RGB+grad3(6ch)
- 盘子权重：利用 depth_raw 估计盘子覆盖率，样本损失按 (1+gamma*coverage) 加权
- 均值启动：最后层 bias=训练集label均值，最后层权重清零
- 头部预热：前若干epoch仅训回归头
- Warmup+Cosine、Huber/MSE 切换、分组学习率、梯度裁剪、EMA
- 支持原生6通道conv1（--native6）或1x1 stem对齐到3通道
- 自动保存 metrics CSV 和训练曲线 PNG
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

    def __len__(self): return len(self.df)

    def _load_rgb_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform_rgb(img)

    def _load_depth_color_tensor(self, path_opt: Optional[str], H: int, W: int) -> torch.Tensor:
        if not self.use_depth_color or not path_opt or (isinstance(path_opt, float) and np.isnan(path_opt)):
            return torch.zeros(3, H, W)
        p = resolve_path(self.images_root, str(path_opt))
        if not p.exists(): return torch.zeros(3, H, W)
        img = Image.open(p).convert("RGB").resize((W, H))
        t = TF.to_tensor(img)
        return (t - 0.5) / 0.25  # 伪彩/深度彩归一化

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
            return TF.to_tensor(arr_img).to(torch.float32)  # [1,H,W]
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
        D, Dx, Dy = d01, _norm01(dx), _norm01(dy)
        feat = torch.cat([D, Dx, Dy], dim=0)  # [3,H,W] in [0,1]
        return (feat - 0.5) / 0.25

    def _plate_mask_from_depth(self, d01: torch.Tensor, k: float = 2.5) -> torch.Tensor:
        """median+MAD 拟合盘面/桌面，返回 [1,H,W] 0/1 mask"""
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
        # 解析RGB
        if "rgb_path" in row and not pd.isna(row["rgb_path"]): rgb_col = "rgb_path"
        elif "image_path" in row and not pd.isna(row["image_path"]): rgb_col = "image_path"
        else: raise KeyError("Row missing 'rgb_path' or 'image_path'.")

        rgb_path = resolve_path(self.images_root, str(row[rgb_col]))
        if not rgb_path.exists(): raise FileNotFoundError(f"RGB not found: {rgb_path}")

        img_rgb = Image.open(rgb_path).convert("RGB")

        # 统一几何增强
        if self.split == "train":
            i,j,h,w = T.RandomResizedCrop.get_params(img_rgb, scale=(0.8,1.0), ratio=(3/4,4/3))
            do_flip = random.random() < 0.5
            rgb_aug = TF.resized_crop(img_rgb, i,j,h,w, (self.img_size,self.img_size), interpolation=IM.BILINEAR)
            if do_flip: rgb_aug = TF.hflip(rgb_aug)
        else:
            rgb_aug = TF.center_crop(TF.resize(img_rgb, (self.img_size,self.img_size), interpolation=IM.BILINEAR),
                                     [self.img_size,self.img_size])

        x_rgb = TF.normalize(TF.to_tensor(rgb_aug), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        C,H,W = x_rgb.shape
        channels = [x_rgb]

        # depth_color
        if self.use_depth_color:
            t_dcol = self._load_depth_color_tensor(row.get("depth_color_path", None), H, W)
            channels.append(t_dcol)

        # depth_raw & grad3
        d01 = torch.zeros(1, H, W)
        if ("depth_raw_path" in row) and not pd.isna(row["depth_raw_path"]):
            # 几何对齐到目标尺寸（与RGB一致）
            p = resolve_path(self.images_root, str(row["depth_raw_path"]))
            if p.exists():
                try:
                    # 用 0-1 深度图
                    d01 = self._load_depth_raw_01_tensor(row["depth_raw_path"], H, W)
                except:
                    d01 = torch.zeros(1, H, W)

        if self.use_depth_raw:
            channels.append((d01 - 0.5) / 0.25)

        if self.use_depth_grad3:
            channels.append(self._depth_grad3(d01))

        x = torch.cat(channels, dim=0)  # [C_total,H,W]

        # 样本权重（盘子覆盖率）
        sample_weight = 1.0
        if self.plate_weight_gamma > 0 and (d01 is not None):
            m_plate = self._plate_mask_from_depth(d01)
            coverage = float(m_plate.mean().item())
            sample_weight = 1.0 + self.plate_weight_gamma * coverage
        sample_weight = torch.tensor([sample_weight], dtype=torch.float32)

        if "label" in self.df.columns and self.split in ("train","val"):
            y = torch.tensor([float(row["label"])], dtype=torch.float32)
            sid = str(row.get("dish_id", idx))
            return x, y, sid, sample_weight
        else:
            sid = str(row.get("ID", row.get("dish_id", idx)))
            return x, sid

# ---------------------------
# Model
# ---------------------------
class ResNetRegressor(nn.Module):
    def __init__(self, in_ch=3, backbone="resnet18", pretrained=True,
                 native6=False, mlp_hidden: Optional[List[int]]=None,
                 mlp_dropout: float=0.0, mlp_act: str="relu"):
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

        # 输入处理
        if native6 and in_ch != 3:
            old = self.backbone.conv1
            new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                new.weight[:, :3] = old.weight
                if in_ch > 3:
                    mean_w = old.weight.mean(dim=1, keepdim=True)
                    new.weight[:, 3:] = mean_w.repeat(1, in_ch-3, 1, 1)
            self.backbone.conv1 = new
            self.stem = None
        else:
            self.stem = (nn.Sequential(
                nn.Conv2d(in_ch, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            ) if in_ch != 3 else None)

        # 去掉 fc，用自定义 head
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

    # 均值启动：最后一层 bias=label_mean，最后一层 weight=0
    def init_output_bias(self, label_mean: float):
        last = None
        for m in reversed(self.head):
            if isinstance(m, nn.Linear):
                last = m; break
        if last is not None:
            nn.init.constant_(last.bias, float(label_mean))
            nn.init.zeros_(last.weight)

# ---------------------------
# EMA
# ---------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k,p in model.state_dict().items() if p.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if k in self.shadow and p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {k: p.clone() for k,p in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow})

    @torch.no_grad()
    def restore(self, model: nn.Module):
        model.load_state_dict({**model.state_dict(), **self.backup})

# ---------------------------
# Loss / Train / Val
# ---------------------------
def weighted_huber(pred, target, weight, delta: float):
    # pred,target:[B,1], weight:[B,1]
    diff = pred - target
    abs_diff = diff.abs()
    loss = torch.where(abs_diff <= delta,
                       0.5 * diff * diff,
                       delta * (abs_diff - 0.5 * delta))
    return (loss * weight).sum() / (weight.sum() + 1e-8)

def weighted_mse(pred, target, weight):
    se = (pred - target) ** 2
    return (se * weight).sum() / (weight.sum() + 1e-8)

def compute_unweighted_metrics(pred, target):
    se = (pred - target) ** 2
    ae = (pred - target).abs()
    return se.sum().item(), ae.sum().item()  # for averaging by n later

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def build_optimizer(model, base_lr, weight_decay, head_lr_mult=1.0, only_head=False):
    head_params = list(model.head.parameters())
    stem_params = list(model.stem.parameters()) if model.stem is not None else []
    backbone_params = [p for n,p in model.named_parameters()
                       if (("head" not in n) and ("stem" not in n))]

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
        sched = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        sched = cosine
    return sched

def train_one_epoch(model, loader, optimizer, device, loss_type, huber_delta, grad_clip, ema: Optional[EMA]):
    model.train()
    mse_sum=0.0; mae_sum=0.0; n=0
    for batch in loader:
        x,y,_,w = batch
        x,y,w = x.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        p = model(x)

        if loss_type == "huber":
            loss = weighted_huber(p, y, w, huber_delta)
        else:
            loss = weighted_mse(p, y, w)

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None: ema.update(model)

        se_sum, ae_sum = compute_unweighted_metrics(p, y)
        mse_sum += se_sum; mae_sum += ae_sum; n += x.size(0)

    mse = mse_sum / max(1,n)
    rmse = math.sqrt(max(0.0, mse))
    mae = mae_sum / max(1,n)
    return mse, rmse, mae

@torch.no_grad()
def validate(model, loader, device, ema: Optional[EMA]):
    model.eval()
    use_ema = ema is not None
    if use_ema: ema.apply_to(model)
    mse_sum=0.0; mae_sum=0.0; n=0
    for batch in loader:
        x,y,_,_ = batch  # 验证不需要权重来计算指标（与历史一致）
        x,y = x.to(device), y.to(device)
        p = model(x)
        se_sum, ae_sum = compute_unweighted_metrics(p, y)
        mse_sum += se_sum; mae_sum += ae_sum; n += x.size(0)
    if use_ema: ema.restore(model)
    mse = mse_sum / max(1,n)
    rmse = math.sqrt(max(0.0, mse))
    mae = mae_sum / max(1,n)
    return mse, rmse, mae

@torch.no_grad()
def predict(model, loader, device) -> List[Tuple[str, float]]:
    model.eval()
    out=[]
    for batch in loader:
        if len(batch)==2:
            x,ids = batch
        else:
            x,_,ids,_ = batch
        x = x.to(device)
        y = model(x).squeeze(1).cpu().numpy().tolist()
        out += list(zip(list(map(str, ids)), list(map(float, y))))
    return out

@torch.no_grad()
def offline_eval_on_val(model, dl_val, device, out_csv_path, ema: Optional[EMA]):
    model.eval()
    ids,preds,gts = [],[],[]
    use_ema = ema is not None
    if use_ema: ema.apply_to(model)
    for batch in dl_val:
        x,y,sid,_ = batch
        x = x.to(device)
        p = model(x).squeeze(1).cpu().numpy()
        y = y.squeeze(1).cpu().numpy()
        ids += list(map(str, sid)); preds += p.tolist(); gts += y.tolist()
    if use_ema: ema.restore(model)
    df = pd.DataFrame({"ID": ids, "pred": preds, "label": gts})
    mse = float(np.mean((df["pred"]-df["label"])**2))
    mae = float(np.mean(np.abs(df["pred"]-df["label"])))
    rmse = math.sqrt(mse)
    print(f"[OFFLINE VAL] N={len(df)} MSE={mse:.4f} RMSE={rmse:.4f} MAE={mae:.4f}")
    df.to_csv(out_csv_path, index=False)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv",  type=str, default=None)
    parser.add_argument("--images_root", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)

    # modalities
    parser.add_argument("--use_depth_color", action="store_true")
    parser.add_argument("--use_depth_raw", action="store_true")
    parser.add_argument("--use_depth_grad3", action="store_true")
    parser.add_argument("--plate_weight_gamma", type=float, default=1.0)

    # model
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","resnet50"])
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--native6", action="store_true")
    parser.add_argument("--mlp_hidden", type=str, default="256,64")
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--mlp_act", type=str, default="gelu", choices=["relu","gelu","silu"])

    # train
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--head_warmup_epochs", type=int, default=2)
    parser.add_argument("--head_lr_mult", type=float, default=5.0)
    parser.add_argument("--loss", type=str, default="huber", choices=["mse","huber"])
    parser.add_argument("--huber_delta", type=float, default=50.0)
    parser.add_argument("--ema", type=float, default=0.0, help="0=off, else decay e.g. 0.995")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)


    # output
    parser.add_argument("--out_dir", type=str, default="outputs_2.0")
    parser.add_argument("--pred_csv", type=str, default="predictions.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    images_root = Path(args.images_root).resolve() if args.images_root else None

    # load dataframes
    train_df = pd.read_csv(args.train_csv)
    needed_cols = {"dish_id","label"}
    if not needed_cols.issubset(set(train_df.columns)):
        raise ValueError(f"train_index.csv must contain {needed_cols} plus path column rgb_path or image_path.")
    label_mean = float(train_df["label"].mean())  # 均值启动用

    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_total = len(train_df); n_val = max(1, int(n_total * args.val_split))
    df_val = train_df.iloc[:n_val].reset_index(drop=True)
    df_trn = train_df.iloc[n_val:].reset_index(drop=True)

    ds_trn = IndexDishDataset(df_trn, images_root, "train", img_size=args.img_size,
                              use_depth_color=args.use_depth_color,
                              use_depth_raw=args.use_depth_raw,
                              use_depth_grad3=args.use_depth_grad3,
                              plate_weight_gamma=args.plate_weight_gamma)
    ds_val = IndexDishDataset(df_val, images_root, "val", img_size=args.img_size,
                              use_depth_color=args.use_depth_color,
                              use_depth_raw=args.use_depth_raw,
                              use_depth_grad3=args.use_depth_grad3,
                              plate_weight_gamma=args.plate_weight_gamma)

    probe = ds_trn[0][0] if isinstance(ds_trn[0], (tuple,list)) else ds_trn[0]
    in_ch = int(probe.shape[0])

    pin_memory = torch.cuda.is_available()
    dl_trn = DataLoader(ds_trn, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=pin_memory)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=pin_memory)

    print(f"Train size: {len(ds_trn)} | Val size: {len(ds_val)} | in_ch: {in_ch} | device: {device}")

    # model
    hidden = [int(x) for x in args.mlp_hidden.split(",") if x.strip()] if args.mlp_hidden else []
    model = ResNetRegressor(in_ch=in_ch, backbone=args.backbone,
                            pretrained=(not args.no_pretrained),
                            native6=args.native6,
                            mlp_hidden=hidden, mlp_dropout=args.mlp_dropout,
                            mlp_act=args.mlp_act).to(device)
    # 均值启动
    model.init_output_bias(label_mean)

    # optim & sched
    # 1) 头部预热
    set_requires_grad(model.backbone, False)
    if model.stem is not None: set_requires_grad(model.stem, False)
    set_requires_grad(model.head, True)
    opt = build_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay,
                          head_lr_mult=args.head_lr_mult, only_head=True)
    sch = build_scheduler(opt, args.epochs, args.warmup_epochs)

    ema = EMA(model, args.ema) if args.ema and args.ema>0 else None

    best_rmse = float("inf"); best_path = out_dir / "best_model.pth"
    hist = {"epoch": [], "train_mse": [], "val_mse": []}

    for ep in range(1, args.epochs+1):
        # 到点解冻：全量训练
        if ep == args.head_warmup_epochs + 1 and any([not p.requires_grad for p in model.backbone.parameters()]):
            set_requires_grad(model.backbone, True)
            if model.stem is not None: set_requires_grad(model.stem, True)
            set_requires_grad(model.head, True)
            opt = build_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay,
                                  head_lr_mult=args.head_lr_mult, only_head=False)
            sch = build_scheduler(opt, args.epochs, args.warmup_epochs)

        tr_mse, tr_rmse, tr_mae = train_one_epoch(model, dl_trn, opt, device,
                                                  loss_type=args.loss, huber_delta=args.huber_delta,
                                                  grad_clip=args.grad_clip, ema=ema)
        va_mse, va_rmse, va_mae = validate(model, dl_val, device, ema=ema)
        sch.step()

        print(f"Epoch {ep:03d}/{args.epochs} | TR MSE: {tr_mse:.4f} RMSE: {tr_rmse:.4f} MAE: {tr_mae:.4f} | "
              f"VA MSE: {va_mse:.4f} RMSE: {va_rmse:.4f} MAE: {va_mae:.4f}")

        hist["epoch"].append(ep); hist["train_mse"].append(tr_mse); hist["val_mse"].append(va_mse)

        if va_rmse < best_rmse:
            best_rmse = va_rmse
            torch.save({"model_state": model.state_dict(),
                        "ema_state": (ema.shadow if ema is not None else None),
                        "in_ch": in_ch, "backbone": args.backbone, "img_size": args.img_size},
                       best_path)
            print(f"  ↳ Saved best: {best_path} (val RMSE {best_rmse:.4f})")
            offline_eval_on_val(model, dl_val, device, out_dir/"val_preds_debug.csv", ema=ema)

    # 保存曲线
    metrics_csv = out_dir / "metrics_trainval.csv"
    pd.DataFrame(hist).to_csv(metrics_csv, index=False)
    print(f"[OK] metrics CSV -> {metrics_csv}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.plot(hist["epoch"], hist["train_mse"], label="Train MSE")
        plt.plot(hist["epoch"], hist["val_mse"], label="Val MSE")
        plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)")
        plt.title("Resnet_DA_Min (RGB-D)")
        plt.legend(); plt.tight_layout()
        fig_path = out_dir / "train_val_mse.png"
        plt.savefig(fig_path, dpi=160)
        print(f"[OK] figure      -> {fig_path}")
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    # 推理（用 best+EMA）
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv)
        if ("image_path" not in test_df.columns) and ("rgb_path" not in test_df.columns):
            raise ValueError("test_index.csv must include 'image_path' or 'rgb_path'.")
        ds_tst = IndexDishDataset(test_df, images_root, "test", img_size=args.img_size,
                                  use_depth_color=args.use_depth_color,
                                  use_depth_raw=args.use_depth_raw,
                                  use_depth_grad3=args.use_depth_grad3,
                                  plate_weight_gamma=args.plate_weight_gamma)
        dl_tst = DataLoader(ds_tst, batch_size=32, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)

        # 加载最佳（含EMA）
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            if ckpt.get("ema_state", None) is not None:
                ema = EMA(model, decay=args.ema if args.ema>0 else 0.995)
                ema.shadow = {k: v.to(device) for k,v in ckpt["ema_state"].items()}
                ema.apply_to(model)

        preds = predict(model, dl_tst, device)
        sub = pd.DataFrame({"ID": [i for i,_ in preds], "Value": [v for _,v in preds]})
        out_csv = out_dir / args.pred_csv
        sub.to_csv(out_csv, index=False)
        print(f"[OK] Wrote predictions to: {out_csv}")

if __name__ == "__main__":
    main()
