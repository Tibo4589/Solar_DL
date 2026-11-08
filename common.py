"""
Shared plumbing for the rooftop solar panel project (Deep Learning course, KU Leuven).

Same set-up as the labs: every numbered script is a standalone program. You run it,
it reads what the previous step left on disk, does ONE job, and writes its output to
disk. Nothing is kept in memory between scripts, so state is always a file.

Difference with the labs: there are two of you. All DATA and OUTPUTS therefore live
in one shared project folder, SOLAR_ROOT (see setup_vsc.sh), so you both see the same
split, the same trained runs and the same plots. Only the CODE lives in your Git repo.

    $SOLAR_ROOT/
      data/bdappv/            BDAPPV, unzipped so that ign/img, ign/mask and metadata.csv exist
      data/flanders/          orthophotos, boundaries, buildings, installation data, labels
      processed/              inventory, split, area factors (steps 01-03)
      runs/<run_name>/        one folder per training run: weights, config, history, curves
      runs/results.csv        one line per finished run (the experiment log you both share)
      outputs/                test metrics, Flemish predictions, area statistics
      plots/                  every figure the scripts make (poster material)

This module holds no decisions of its own except the SETTINGS block below, which you
edit as you make decisions (exactly like the settings block in lab 1's common.py).
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # no display on a cluster node: always save PNGs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOLAR_ROOT = Path(os.environ.get("SOLAR_ROOT", Path(__file__).resolve().parent / "project_data"))

DATA_DIR = SOLAR_ROOT / "data"
BDAPPV_DIR = DATA_DIR / "bdappv"
CAMPAIGN = "ign"                          # we train on the IGN images (0.20 m/pixel), not Google
IMG_DIR = BDAPPV_DIR / CAMPAIGN / "img"
MASK_DIR = BDAPPV_DIR / CAMPAIGN / "mask"
METADATA_CSV = BDAPPV_DIR / "metadata.csv"
FLANDERS_DIR = DATA_DIR / "flanders"
LABELS_DIR = FLANDERS_DIR / "labels"      # chips to label + your QGIS polygons (steps 10-11)

PROCESSED_DIR = SOLAR_ROOT / "Solar_DL/processed"
RUNS_DIR = SOLAR_ROOT / "Solar_DL/runs"
RESULTS_CSV = RUNS_DIR / "results.csv"
OUTPUTS_DIR = SOLAR_ROOT / "Solar_DL/outputs"
PLOTS_DIR = SOLAR_ROOT / "Solar_DL/plots"

# ---------------------------------------------------------------------------
# SETTINGS: edit these as you make decisions (write down WHY in your log)
# ---------------------------------------------------------------------------

SEED = 42
GSD = 0.20            # metres per pixel. IGN thumbnails are 0.20 m; every other image is resampled to this.
TILE = 384            # training crop (pixels). Multiple of 32: the encoders downsample 5 times (2^5 = 32).
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-4             # Adam(W) learning rate for a PRETRAINED encoder: small, so ImageNet features are not destroyed
WEIGHT_DECAY = 1e-4   # L2-style regularisation (AdamW applies it decoupled from the gradient)
PATIENCE = 8          # early stopping: stop when val IoU has not improved for this many epochs
NUM_WORKERS = 4       # CPU processes that load/augment images while the GPU trains
LOSS = "bce_dice"     # "bce", "dice" or "bce_dice" (step 05 experiments decide)
INCLUDE_UNMASKED = False   # step 03: use BDAPPV images WITHOUT a mask as negatives? Only after checking them (step 02)
MIN_PANEL_AREA_M2 = 1.0    # steps 13-14: blobs smaller than this are noise, not a panel (one module is ~1.7-2 m²)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, RUNS_DIR, OUTPUTS_DIR, PLOTS_DIR, LABELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print('Asked for "cuda" but there is no GPU here: falling back to the CPU (full training will be very slow).')
        return torch.device("cpu")
    return torch.device(requested)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# ---------------------------------------------------------------------------
# Reading images and masks
# ---------------------------------------------------------------------------

def read_rgb(path, scale: float = 1.0) -> np.ndarray:
    """uint8 (H, W, 3). scale < 1 shrinks the image (e.g. 0.5 turns a 0.10 m image into 0.20 m)."""
    img = Image.open(path).convert("RGB")
    if scale != 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
    return np.asarray(img)


def read_mask(path, shape=None, scale: float = 1.0) -> np.ndarray:
    """uint8 (H, W) with values 0/1. path None means 'no panel in this image' (an all-zero mask)."""
    if path is None or (isinstance(path, float) and np.isnan(path)) or str(path) in ("", "nan", "None"):
        return np.zeros(shape, dtype=np.uint8)
    m = Image.open(path).convert("L")
    if scale != 1.0:
        # NEAREST: a mask holds classes, not intensities; bilinear would invent values between 0 and 1
        m = m.resize((round(m.width * scale), round(m.height * scale)), Image.NEAREST)
    return (np.asarray(m) > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegDataset(Dataset):
    """PyTorch Dataset: returns one (image, mask) pair per index.

    image: float (3, H, W) in [0, 1]      mask: float (1, H, W), 1 = solar panel pixel
    Normalisation with the ImageNet statistics is the first thing the MODEL does
    (models.SegModel), exactly as in lab 2, so the Dataset stays model-independent.

    augment=True (training only):
      - random TILE x TILE crop  (also gives the network differently-placed panels)
      - random rotation by k*90 degrees and mirroring: a roof seen from above has no
        'up' direction, so these never create unrealistic images (as for the hazelnuts in lab 3)
      - small brightness/contrast/colour jitter: the Flemish winter photos are darker
        and differently coloured than the French ones; jitter makes the model less
        sensitive to that (it does NOT remove the domain shift, only softens it)
    """

    def __init__(self, img_paths, mask_paths, augment: bool = False, scale: float = 1.0, crop: int = TILE):
        assert len(img_paths) == len(mask_paths)
        self.img_paths = list(img_paths)
        self.mask_paths = list(mask_paths)
        self.augment = augment
        self.scale = scale
        self.crop = crop

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = read_rgb(self.img_paths[idx], self.scale)
        mask = read_mask(self.mask_paths[idx], img.shape[:2], self.scale)
        img = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0   # (3, H, W)
        mask = torch.from_numpy(mask.copy())[None].float()                     # (1, H, W)
        if self.augment:
            img, mask = self._augment(img, mask)
        return img, mask

    def _augment(self, img, mask):
        _, h, w = img.shape
        c = self.crop
        if h < c or w < c:  # small images (e.g. after resampling): pad to the crop size
            ph, pw = max(0, c - h), max(0, c - w)
            img = F.pad(img, (0, pw, 0, ph))
            mask = F.pad(mask, (0, pw, 0, ph))
            _, h, w = img.shape
        top = int(torch.randint(0, h - c + 1, ()))
        left = int(torch.randint(0, w - c + 1, ()))
        img, mask = img[:, top:top + c, left:left + c], mask[:, top:top + c, left:left + c]
        k = int(torch.randint(4, ()))
        img, mask = torch.rot90(img, k, dims=(1, 2)), torch.rot90(mask, k, dims=(1, 2))
        if torch.rand(()) < 0.5:
            img, mask = img.flip(-1), mask.flip(-1)
        # photometric jitter on the image only (the mask does not change when the light changes)
        brightness = 1.0 + 0.2 * (torch.rand(()) * 2 - 1)
        contrast = 1.0 + 0.2 * (torch.rand(()) * 2 - 1)
        colour = 1.0 + 0.05 * (torch.rand(3, 1, 1) * 2 - 1)
        mean = img.mean()
        img = ((img - mean) * contrast + mean) * brightness * colour
        return img.clamp(0, 1), mask


def load_split(split_file: str = "split.csv"):
    import pandas as pd
    path = PROCESSED_DIR / split_file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 03_make_split.py first.")
    df = pd.read_csv(path, dtype={"stem": str, "group": str})
    df["mask_path"] = df["mask_path"].where(df["mask_path"].notna(), None)
    return df


def make_datasets(frac: float = 1.0, seed: int = SEED, split_file: str = "split.csv"):
    """Train/val/test SegDatasets from processed/split.csv. frac < 1 keeps only that
    fraction of the training images (used for the learning curve, step 06)."""
    df = load_split(split_file)
    tr = df[df.split == "train"]
    if frac < 1.0:
        tr = tr.sample(frac=frac, random_state=seed)
    va, te = df[df.split == "val"], df[df.split == "test"]
    return (SegDataset(tr.img_path, tr.mask_path, augment=True),
            SegDataset(va.img_path, va.mask_path, augment=False),
            SegDataset(te.img_path, te.mask_path, augment=False))


def make_loader(ds, batch_size: int = BATCH_SIZE, shuffle: bool = False, num_workers: int = NUM_WORKERS):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss = 1 - 2|P∩T| / (|P| + |T|), computed on probabilities over the whole batch.

    Why: in our images only a few % of the pixels are panel. Plain BCE can reach a low
    loss by predicting 'background' everywhere. Dice only looks at the overlap with the
    panel pixels, so it cannot be fooled by the huge number of easy background pixels.
    eps keeps the loss defined for batches without any panel."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


class SegLoss(nn.Module):
    def __init__(self, kind: str = LOSS):
        super().__init__()
        assert kind in ("bce", "dice", "bce_dice"), kind
        self.kind = kind
        self.bce = nn.BCEWithLogitsLoss()   # sigmoid + binary cross-entropy in one numerically stable op (lab 1, step 06)

    def forward(self, logits, target):
        if self.kind == "bce":
            return self.bce(logits, target)
        if self.kind == "dice":
            return dice_loss(logits, target)
        return 0.5 * self.bce(logits, target) + 0.5 * dice_loss(logits, target)


# ---------------------------------------------------------------------------
# Metrics: counted over PIXELS, pooled over the whole set
# ---------------------------------------------------------------------------

@dataclass
class Counts:
    """Confusion-matrix counts over pixels. Pooled over all images (not averaged per image):
    an image without panels has no defined IoU, and pooling weighs every pixel equally."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = pred.bool(), target.bool()
        self.tp += int((pred & target).sum())
        self.fp += int((pred & ~target).sum())
        self.fn += int((~pred & target).sum())
        self.tn += int((~pred & ~target).sum())

    def metrics(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        iou = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)            # = Dice coefficient
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)  # reported only to show why it is useless here
        area_ratio = (tp + fp) / max(tp + fn, 1)          # predicted panel area / true panel area
        return {"iou": iou, "f1": f1, "precision": precision, "recall": recall,
                "accuracy": accuracy, "area_ratio": area_ratio}


# ---------------------------------------------------------------------------
# Training loop (same structure as labs 1-2, with masks instead of labels)
# ---------------------------------------------------------------------------

def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, threshold: float = 0.5):
    """One pass over a DataLoader. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, counts = 0.0, Counts()
    use_amp = device.type == "cuda"
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            # mixed precision on the GPU: ~2x faster, half the memory; the loss is still computed safely
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits.float(), masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item()
            counts.add(torch.sigmoid(logits.float()) > threshold, masks > 0.5)
    return total_loss / max(len(loader), 1), counts.metrics()


def train_model(model, train_loader, val_loader, optimizer, device, run_dir: Path,
                epochs: int = EPOCHS, loss_kind: str = LOSS, patience: int = PATIENCE, desc: str = "training"):
    """Epochs x mini-batches, with early stopping on the validation IoU.

    The weights of the best validation epoch are saved to run_dir/best.pt and loaded back
    at the end (lab 1, step 16: 'stop at the best epoch'). We select on val IoU rather than
    val loss (lab 1) because IoU is the quantity we report; the loss mixes BCE and Dice and
    can keep falling while the IoU at our threshold no longer improves."""
    loss_fn = SegLoss(loss_kind)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # learning-rate decay
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [], "val_f1": [], "lr": []}
    best_iou, waited = -1.0, 0
    for epoch in tqdm(range(1, epochs + 1), desc=desc, unit="epoch"):
        tr_loss, tr_m = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler)
        va_loss, va_m = run_epoch(model, val_loader, loss_fn, device)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_iou"].append(tr_m["iou"])
        history["val_iou"].append(va_m["iou"])
        history["val_f1"].append(va_m["f1"])
        tqdm.write(f"Epoch {epoch:02d}/{epochs} | train loss {tr_loss:.4f} IoU {tr_m['iou']:.3f} | "
                   f"val loss {va_loss:.4f} IoU {va_m['iou']:.3f} P {va_m['precision']:.3f} R {va_m['recall']:.3f}")
        if va_m["iou"] > best_iou:
            best_iou, waited = va_m["iou"], 0
            history["best_epoch"], history["best_val"] = epoch, va_m
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            waited += 1
            if waited >= patience:
                tqdm.write(f"Early stopping: no val IoU improvement for {patience} epochs.")
                break
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    return history


# ---------------------------------------------------------------------------
# Runs: saving, loading, the shared results log
# ---------------------------------------------------------------------------

def save_run(run_dir: Path, config: dict, history: dict) -> None:
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


def load_run(name: str, device=torch.device("cpu")):
    """(model, config, history) of a finished run. The model is rebuilt from config.json
    and gets the weights of its best validation epoch (best.pt)."""
    from models import build_model
    run_dir = RUNS_DIR / name
    if not (run_dir / "best.pt").exists():
        raise FileNotFoundError(f"{run_dir / 'best.pt'} not found. Train it first (05_train.py --run-name {name}).")
    config = json.loads((run_dir / "config.json").read_text())
    history = json.loads((run_dir / "history.json").read_text())
    model = build_model(config["model"], config.get("encoder"), pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    return model.to(device).eval(), config, history


def run_threshold(name: str) -> float:
    """The threshold chosen on the validation set by step 08 (0.5 if step 08 was not run yet)."""
    path = RUNS_DIR / name / "threshold.json"
    if path.exists():
        return float(json.loads(path.read_text())["threshold"])
    print(f"  (no threshold.json for {name}: using 0.5. Run 08_choose_threshold.py --run {name})")
    return 0.5


def append_result(row: dict) -> None:
    """One line per finished run in runs/results.csv: your shared experiment log."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists()
    fields = ["time", "who", "run", "model", "encoder", "loss", "lr", "frac", "freeze_encoder", "pretrained",
              "init_from", "data", "n_train", "epochs_run", "best_epoch", "val_iou", "val_f1",
              "val_precision", "val_recall", "params_M", "train_min"]
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"time": time.strftime("%Y-%m-%d %H:%M"), "who": os.environ.get("USER", "?"), **row})


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_probs(model, images: torch.Tensor, device, batch_size: int = 16) -> torch.Tensor:
    """(N, 3, H, W) images in [0, 1] -> (N, 1, H, W) panel probabilities (on the CPU)."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(x).float()).cpu())
    return torch.cat(out)


def iterate_probs(model, dataset, device, batch_size: int = 8, num_workers: int = NUM_WORKERS):
    """Yields (images, masks, probs) batch by batch over a whole dataset, so a big set never
    has to fit in memory at once."""
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for images, masks in loader:
        yield images, masks, predict_probs(model, images, device, batch_size)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histories(histories: dict, title: str, out_name: str) -> Path:
    """Train loss, val loss and val IoU of each run side by side (dot = best epoch), as in labs 1-2."""
    import matplotlib.pyplot as plt
    ensure_dirs()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for label, h in histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        best = h.get("best_epoch", len(h["train_loss"]))
        axes[0].plot(ep, h["train_loss"], label=label)
        line, = axes[1].plot(ep, h["val_loss"], label=label)
        axes[1].plot(best, h["val_loss"][best - 1], "o", color=line.get_color())
        line, = axes[2].plot(ep, h["val_iou"], label=label)
        axes[2].plot(best, h["val_iou"][best - 1], "o", color=line.get_color())
    for ax, ylabel in zip(axes, ["Training loss", "Validation loss", "Validation IoU"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle(f"{title}  (dot = best validation epoch)")
    plt.tight_layout()
    out = PLOTS_DIR / out_name
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def overlay(img: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    """RGB uint8 image with the mask painted on top in a colour."""
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.astype(np.uint8)


def to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float tensor in [0, 1] -> (H, W, 3) uint8."""
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def error_image(img: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Green = correct panel (TP), yellow = false alarm (FP), cyan = missed panel (FN).
    (Not red: roof tiles are red, a red overlay disappears on them.)"""
    t, p = truth.astype(bool), pred.astype(bool)
    out = overlay(img, t & p, (0, 220, 0), 0.6)
    out = overlay(out, ~t & p, (255, 230, 0), 0.7)
    return overlay(out, t & ~p, (0, 255, 255), 0.7)
