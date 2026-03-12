# vehicle_transformer_v4.py
# ─────────────────────────────────────────────────────────────────────────────
# Improvements over v3:
#   1. Label smoothing (ε=0.1) — prevents overconfident 0/1 outputs
#   2. Temperature scaling — post-hoc calibration of probabilities
#   3. Vehicle augmentation — 5× augment positive samples (noise + time shift)
#   4. Focal loss — focuses learning on hard examples near decision boundary
#   5. More files (800) for more vehicle diversity
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert, medfilt
import re
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE = Path("/home/augustin/Documents/IA_project/Spintronic Sensor Dataset for Vehicle Detection and Car Model Recognition (14 Days, Unlabeled with Timestamps)")
DATA_DIRS = [
    BASE / "Run 20250204 15h24m13s - 20250211 16h30m16s",
    BASE / "Run 20250212 14h41m50s - 20250214 16h30m31s",
    BASE / "Run 20250218 11h12m41s - 20250220 14h40m28s",
]
TEST_DIR   = BASE / "Run 20250224 14h55m20s - 20250227 14h38m27s"
OUTPUT_DIR = BASE / "transformer_output_v4"

# Signal
SAMPLE_RATE   = 2400
CARRIER_HZ    = 24.9
CARRIER_BW    = 5.0
EDGE_TRIM     = int(0.5 * SAMPLE_RATE)
DETREND_K     = int(5.0 * SAMPLE_RATE)
DETREND_K     = DETREND_K if DETREND_K % 2 == 1 else DETREND_K + 1
IQR_CLIP      = 0.5

# Window
SEQ_LEN       = 2400
STEP_SIZE     = 240
CONTEXT_WINS  = 16

# Auto-labeling
VEHICLE_THRESH   = 1.5
VEHICLE_MIN_PTS  = 50

# Architecture
PATCH_SIZE    = 50
D_MODEL       = 128
N_HEADS       = 4
N_LAYERS      = 4
D_FF          = 256
DROPOUT       = 0.15    # slightly higher dropout for better regularization

# Training improvements
LABEL_SMOOTHING = 0.10  # prevents hard 0/1 targets
FOCAL_GAMMA     = 2.0   # focal loss — focus on hard examples
AUG_FACTOR      = 5     # augment each vehicle window N times
AUG_NOISE_STD   = 0.08  # gaussian noise std for augmentation
AUG_SHIFT_MAX   = 24    # max sample shift for time augmentation (0.01s)

# Training
BATCH_SIZE    = 128
EPOCHS        = 40
LR            = 2e-4
VAL_SPLIT     = 0.2
MAX_FILES     = 500     # more files = more vehicle diversity
PATIENCE      = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
_bp_lo = (CARRIER_HZ - CARRIER_BW) / (SAMPLE_RATE / 2)
_bp_hi = (CARRIER_HZ + CARRIER_BW) / (SAMPLE_RATE / 2)
BP_B, BP_A = butter(4, [_bp_lo, _bp_hi], btype='band')
TIMESTAMP_RE = re.compile(r"Start@ (\d{8}) - (\d{2})h(\d{2})m(\d{2})s")

def parse_timestamp(fname):
    m = TIMESTAMP_RE.search(str(fname))
    if not m: return None
    return datetime.strptime(f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}", "%Y%m%d%H%M%S")

def time_features(dt):
    if dt is None: return np.zeros(6, dtype=np.float32)
    h = (dt.hour + dt.minute/60.0) / 24.0
    d = dt.weekday() / 7.0
    return np.array([
        np.sin(2*np.pi*h), np.cos(2*np.pi*h),
        np.sin(2*np.pi*d), np.cos(2*np.pi*d),
        h, float(7 <= dt.hour <= 19)
    ], dtype=np.float32)

def demodulate(raw):
    bp  = filtfilt(BP_B, BP_A, raw.astype(np.float64))
    env = np.abs(hilbert(bp)).astype(np.float32)
    return env[EDGE_TRIM : len(env) - EDGE_TRIM]

def detrend_normalize(env):
    x    = env.astype(np.float64)
    base = medfilt(x, kernel_size=DETREND_K)
    det  = (x - base).astype(np.float32)
    n, half = len(det), DETREND_K // 2
    step    = max(1, DETREND_K // 20)
    anchors = np.arange(0, n, step)
    vals    = np.array([
        max(float(np.subtract(*np.percentile(
            det[max(0,c-half):min(n,c+half+1)],[75,25]))), IQR_CLIP)
        for c in anchors], dtype=np.float32)
    scale = np.interp(np.arange(n), anchors, vals).astype(np.float32)
    return det / scale

def process_file(bsd_path):
    raw    = np.frombuffer(bsd_path.read_bytes(), dtype=np.int16)
    normed = detrend_normalize(demodulate(raw))
    dt     = parse_timestamp(bsd_path.name)
    return normed, dt

def autolabel(window):
    return 1 if int(np.sum(np.abs(window) > VEHICLE_THRESH)) >= VEHICLE_MIN_PTS else 0

# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def augment_context(ctx):
    """
    Apply random augmentation to a (CONTEXT_WINS, SEQ_LEN) context.
    Only applied to vehicle-positive samples during dataset building.
    """
    aug = ctx.copy()

    # 1. Additive Gaussian noise
    aug += np.random.normal(0, AUG_NOISE_STD, aug.shape).astype(np.float32)

    # 2. Random amplitude scaling (±15%)
    scale = np.random.uniform(0.85, 1.15)
    aug  *= scale

    # 3. Random time shift within the window (circular)
    shift = np.random.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1)
    if shift != 0:
        aug = np.roll(aug, shift, axis=1)

    return aug

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET BUILDING
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset(data_dirs, max_files=MAX_FILES):
    all_files = []
    for d in data_dirs:
        all_files.extend(f for f in Path(d).rglob("*.bsd") if "BIN Time" in f.name)
    all_files = sorted(all_files, key=lambda f: f.name)[:max_files]
    print(f"Found {len(all_files):,} BIN Time files (using {min(len(all_files), max_files):,})")

    signals   = []
    timefeats = []
    index     = []    # (sig_idx, win_start, label)
    n_veh     = 0

    for fi, bsd in enumerate(all_files):
        try:
            normed, dt = process_file(bsd)
        except:
            continue

        n_wins = max(0, (len(normed) - SEQ_LEN) // STEP_SIZE + 1)
        if n_wins < CONTEXT_WINS:
            continue

        wins = np.lib.stride_tricks.as_strided(
            normed, shape=(n_wins, SEQ_LEN),
            strides=(normed.strides[0]*STEP_SIZE, normed.strides[0])
        ).copy()

        labels = np.array([autolabel(wins[i]) for i in range(n_wins)], dtype=np.int64)
        tf     = time_features(dt)
        sig_idx = len(signals)
        signals.append(wins)
        timefeats.append(tf)

        for i in range(n_wins - CONTEXT_WINS + 1):
            label = int(labels[i + CONTEXT_WINS - 1])
            index.append((sig_idx, i, label, False))  # False = not augmented
            if label == 1:
                n_veh += 1
                # Add AUG_FACTOR augmented copies of vehicle contexts
                for _ in range(AUG_FACTOR):
                    index.append((sig_idx, i, label, True))  # True = augment on load

        if fi % 200 == 0:
            total = len(index)
            print(f"  [{100*fi/len(all_files):5.1f}%]  file {fi+1}"
                  f"  contexts: {total:,}  vehicle: {n_veh:,}"
                  f" ({100*n_veh/max(sum(1 for x in index if not x[3]),1):.2f}%)")

    y = np.array([x[2] for x in index], dtype=np.int64)
    n_aug = sum(1 for x in index if x[3])
    print(f"\nDataset: {len(index):,} contexts  "
          f"(incl. {n_aug:,} augmented vehicle samples)")
    print(f"  Vehicle    : {(y==1).sum():,}  ({100*y.mean():.2f}%)")
    print(f"  No vehicle : {(y==0).sum():,}  ({100*(y==0).mean():.2f}%)")
    print(f"  Signal RAM : {sum(s.nbytes for s in signals)/1e9:.2f} GB")
    return signals, timefeats, index, y

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class VehicleDataset(Dataset):
    def __init__(self, signals, timefeats, index):
        self.signals   = signals
        self.timefeats = timefeats
        self.index     = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sig_idx, win_start, label, do_aug = self.index[idx]
        ctx = self.signals[sig_idx][win_start : win_start + CONTEXT_WINS].copy()
        if do_aug:
            ctx = augment_context(ctx)
        tf  = self.timefeats[sig_idx]
        x   = torch.from_numpy(ctx[:, :, np.newaxis])
        t   = torch.from_numpy(tf.copy())
        y   = torch.tensor(label, dtype=torch.long)
        return x, t, y


def make_loaders(signals, timefeats, index, y):
    n       = len(index)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_val

    # Important: keep augmented samples in train only, not val
    idx_tr = index[:n_train]
    idx_va = [x for x in index[n_train:] if not x[3]]  # no aug in val
    y_tr   = y[:n_train]
    y_va   = np.array([x[2] for x in idx_va], dtype=np.int64)

    train_ds = VehicleDataset(signals, timefeats, idx_tr)
    val_ds   = VehicleDataset(signals, timefeats, idx_va)

    n_veh   = int((y_tr==1).sum())
    n_noveh = n_train - n_veh
    if n_veh > 0 and n_noveh > 0:
        weights = np.where(y_tr==1, n_train/n_veh, n_train/n_noveh).astype(np.float32)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=2, pin_memory=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=2, pin_memory=False)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2,
                            shuffle=False, num_workers=2, pin_memory=False)

    print(f"Train: {n_train:,}  Val: {len(idx_va):,}")
    print(f"Train veh: {100*y_tr.mean():.2f}%  Val veh: {100*y_va.mean():.2f}%")
    return train_loader, val_loader

# ══════════════════════════════════════════════════════════════════════════════
#  FOCAL LOSS WITH LABEL SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════
class FocalLossWithSmoothing(nn.Module):
    """
    Combines:
    - Label smoothing: soft targets instead of hard 0/1
    - Focal loss: down-weights easy examples, focuses on hard ones
    """
    def __init__(self, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTHING, n_classes=2):
        super().__init__()
        self.gamma     = gamma
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits, targets):
        # Smooth targets: 0 → ε/K,  1 → 1-ε+ε/K
        with torch.no_grad():
            smooth_targets = torch.full_like(
                logits, self.smoothing / self.n_classes
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1),
                                    1.0 - self.smoothing + self.smoothing/self.n_classes)

        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()

        # Focal weight: (1 - p_t)^gamma
        p_t          = (probs * smooth_targets).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy with smooth targets
        ce   = -(smooth_targets * log_probs).sum(dim=1)
        loss = (focal_weight * ce).mean()
        return loss

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL (same architecture as v3)
# ══════════════════════════════════════════════════════════════════════════════
class PatchEmbed1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj      = nn.Linear(PATCH_SIZE, D_MODEL)
        self.n_patches = SEQ_LEN // PATCH_SIZE
    def forward(self, x):
        x = x.squeeze(-1)
        return self.proj(x.reshape(x.shape[0], self.n_patches, PATCH_SIZE))

class TemporalVehicleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        n_patches = SEQ_LEN // PATCH_SIZE
        self.patch_embed = PatchEmbed1D()
        self.patch_pos   = nn.Parameter(torch.randn(1, n_patches, D_MODEL)*0.02)
        self.patch_cls   = nn.Parameter(torch.randn(1, 1, D_MODEL)*0.02)
        pl = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT,
                                        batch_first=True, norm_first=True)
        self.patch_encoder = nn.TransformerEncoder(pl, num_layers=2)
        self.patch_norm    = nn.LayerNorm(D_MODEL)

        self.context_pos   = nn.Parameter(torch.randn(1, CONTEXT_WINS+1, D_MODEL)*0.02)
        self.time_proj     = nn.Linear(6, D_MODEL)
        self.context_cls   = nn.Parameter(torch.randn(1, 1, D_MODEL)*0.02)
        cl = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT,
                                        batch_first=True, norm_first=True)
        self.context_encoder = nn.TransformerEncoder(cl, num_layers=N_LAYERS)
        self.context_norm    = nn.LayerNorm(D_MODEL)

        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL//2), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(D_MODEL//2, 2)
        )

    def encode_window(self, x):
        B = x.shape[0]
        p = self.patch_embed(x) + self.patch_pos
        cls = self.patch_cls.expand(B, -1, -1)
        return self.patch_norm(self.patch_encoder(torch.cat([cls,p],1))[:,0])

    def forward(self, x, t):
        B  = x.shape[0]
        wv = self.encode_window(x.reshape(B*CONTEXT_WINS, SEQ_LEN, 1))
        wv = wv.reshape(B, CONTEXT_WINS, D_MODEL)
        ctx = torch.cat([self.time_proj(t).unsqueeze(1), wv], 1) + self.context_pos
        cls = self.context_cls.expand(B, -1, -1)
        out = self.context_norm(self.context_encoder(torch.cat([cls,ctx],1))[:,0])
        return self.classifier(out)

# ══════════════════════════════════════════════════════════════════════════════
#  TEMPERATURE SCALING (post-hoc calibration)
# ══════════════════════════════════════════════════════════════════════════════
class TemperatureScaler(nn.Module):
    """
    Learns a single scalar T that divides logits: logits / T
    T > 1 → softer probabilities (less confident)
    T < 1 → sharper probabilities (more confident)
    Fitted on validation set after main training.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x, t):
        logits = self.model(x, t)
        return logits / self.temperature

    def fit(self, val_loader, device):
        """Fit temperature on validation set using NLL loss."""
        self.to(device)
        self.model.eval()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)
        nll_criterion = nn.CrossEntropyLoss()

        # Collect all logits and labels first
        all_logits, all_labels = [], []
        with torch.no_grad():
            for X_b, T_b, y_b in val_loader:
                X_b, T_b = X_b.to(device), T_b.to(device)
                logits = self.model(X_b, T_b)
                all_logits.append(logits.cpu())
                all_labels.append(y_b)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        def eval_temp():
            optimizer.zero_grad()
            scaled = all_logits.to(device) / self.temperature
            loss   = nll_criterion(scaled, all_labels.to(device))
            loss.backward()
            return loss

        optimizer.step(eval_temp)
        print(f"  Temperature calibrated: T = {self.temperature.item():.4f}")
        print(f"  (T>1 = softer, T<1 = sharper)")
        return self.temperature.item()

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for X_b, T_b, y_b in loader:
        X_b = X_b.to(DEVICE, non_blocking=True)
        T_b = T_b.to(DEVICE, non_blocking=True)
        y_b = y_b.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16,
                                enabled=(DEVICE=="cuda")):
            logits = model(X_b, T_b)
            loss   = criterion(logits, y_b)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(y_b)
        correct  += (logits.argmax(1) == y_b).sum().item()
        total    += len(y_b)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    ce = nn.CrossEntropyLoss()
    all_p, all_y, all_prob = [], [], []
    for X_b, T_b, y_b in loader:
        X_b, T_b, y_b = X_b.to(DEVICE), T_b.to(DEVICE), y_b.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16,
                                enabled=(DEVICE=="cuda")):
            logits = model(X_b, T_b)
        loss_sum += ce(logits, y_b).item() * len(y_b)
        preds     = logits.argmax(1)
        correct  += (preds == y_b).sum().item()
        total    += len(y_b)
        all_p.append(preds.cpu())
        all_y.append(y_b.cpu())
        all_prob.append(F.softmax(logits, dim=1)[:,1].cpu())

    all_p    = torch.cat(all_p)
    all_y    = torch.cat(all_y)
    all_prob = torch.cat(all_prob).numpy()

    tp = ((all_p==1)&(all_y==1)).sum().item()
    fp = ((all_p==1)&(all_y==0)).sum().item()
    fn = ((all_p==0)&(all_y==1)).sum().item()
    prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
    f1   = 2*prec*rec/(prec+rec+1e-8)

    # Calibration: fraction of predictions in each prob bucket
    prob_std = float(np.std(all_prob))
    return loss_sum/total, correct/total, prec, rec, f1, prob_std

def train(model, train_loader, val_loader):
    criterion  = FocalLossWithSmoothing()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler     = torch.amp.GradScaler(enabled=(DEVICE=="cuda"))
    best_f1    = 0.0
    no_improve = 0
    best_path  = OUTPUT_DIR / "best_model_v4.pt"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\nTraining on {DEVICE}  "
          f"(focal γ={FOCAL_GAMMA}, label_smooth={LABEL_SMOOTHING}, "
          f"aug×{AUG_FACTOR}, patience={PATIENCE})")
    print(f"{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} "
          f"{'VaAcc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ProbStd':>8}")
    print("─" * 80)

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc                          = train_epoch(model, train_loader,
                                                               optimizer, criterion, scaler)
        va_loss, va_acc, prec, rec, f1, prob_std = eval_epoch(model, val_loader)
        scheduler.step()

        marker = " ← best" if f1 > best_f1 else ""
        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_acc:>6.1%}  "
              f"{va_loss:>8.4f}  {va_acc:>6.1%}  "
              f"{prec:>6.1%}  {rec:>6.1%}  {f1:>6.1%}  {prob_std:>8.4f}{marker}")

        if f1 > best_f1:
            best_f1 = f1; no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "f1": f1, "config": {
                            "SEQ_LEN": SEQ_LEN, "CONTEXT_WINS": CONTEXT_WINS,
                            "PATCH_SIZE": PATCH_SIZE, "D_MODEL": D_MODEL}
                        }, best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nBest F1: {best_f1:.1%}  →  {best_path}")
    return best_f1

# ══════════════════════════════════════════════════════════════════════════════
#  TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_test(model, test_dir, max_files=400):
    model.eval()
    test_files = sorted(
        (f for f in Path(test_dir).rglob("*.bsd") if "BIN Time" in f.name),
        key=lambda f: f.name)[:max_files]
    print(f"Test files: {len(test_files):,}")

    tp=fp=fn=tn=0
    all_probs = []

    for bsd in test_files:
        try: normed, dt = process_file(bsd)
        except: continue

        n_wins = max(0, (len(normed)-SEQ_LEN)//STEP_SIZE+1)
        if n_wins < CONTEXT_WINS: continue

        wins = np.lib.stride_tricks.as_strided(
            normed, shape=(n_wins, SEQ_LEN),
            strides=(normed.strides[0]*STEP_SIZE, normed.strides[0])).copy()
        labels = np.array([autolabel(wins[i]) for i in range(n_wins)])
        tf     = time_features(dt)
        buf    = np.zeros((CONTEXT_WINS, SEQ_LEN), dtype=np.float32)

        for i in range(n_wins):
            buf    = np.roll(buf, -1, axis=0); buf[-1] = wins[i]
            if i < CONTEXT_WINS-1: continue
            label  = int(labels[i])
            x_t    = torch.from_numpy(buf[np.newaxis,:,:,np.newaxis]).to(DEVICE)
            tf_t   = torch.from_numpy(tf[np.newaxis]).to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE=="cuda")):
                logits = model(x_t, tf_t)
            prob = float(F.softmax(logits, dim=1)[0,1].cpu())
            pred = int(prob > 0.5)
            all_probs.append(prob)
            if pred==1 and label==1: tp+=1
            elif pred==1 and label==0: fp+=1
            elif pred==0 and label==1: fn+=1
            else: tn+=1

    prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
    f1   = 2*prec*rec/(prec+rec+1e-8)
    acc  = (tp+tn)/(tp+fp+fn+tn+1e-8)
    all_probs = np.array(all_probs)
    print(f"  Accuracy  : {acc:.1%}")
    print(f"  Precision : {prec:.1%}  Recall: {rec:.1%}  F1: {f1:.1%}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Prob std  : {all_probs.std():.4f}  "
          f"(higher = better calibrated, v3 was ≈0.0)")
    return f1

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*65)
    print("Temporal Vehicle Transformer v4 — Calibrated")
    print(f"Device          : {DEVICE}")
    print(f"Improvements    : focal loss + label smoothing + augmentation")
    print(f"Files           : {MAX_FILES} (vs 400 in v3)")
    print(f"Augmentation    : ×{AUG_FACTOR} vehicle samples")
    print("="*65)

    signals, timefeats, index, y = build_dataset(DATA_DIRS, max_files=MAX_FILES)
    train_loader, val_loader     = make_loaders(signals, timefeats, index, y)

    model   = TemporalVehicleTransformer().to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {n_param:,}  ({n_param/1e6:.2f}M)")

    train(model, train_loader, val_loader)

    print("\nLoading best model...")
    ckpt = torch.load(OUTPUT_DIR/"best_model_v4.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # ── Temperature scaling ──────────────────────────────────────────────────
    print("\nFitting temperature scaling on validation set...")
    ts_model = TemperatureScaler(model)
    T_val    = ts_model.fit(val_loader, DEVICE)
    torch.save({"model_state": model.state_dict(), "temperature": T_val,
                "f1": ckpt["f1"], "epoch": ckpt["epoch"],
                "config": ckpt["config"]},
               OUTPUT_DIR / "best_model_v4_calibrated.pt")
    print(f"Calibrated model saved.")

    print(f"\n{'='*65}")
    print("Validation (with temperature scaling):")
    _, _, prec, rec, f1, pstd = eval_epoch(ts_model, val_loader)
    print(f"  Precision: {prec:.1%}  Recall: {rec:.1%}  F1: {f1:.1%}  ProbStd: {pstd:.4f}")

    print(f"\n{'='*65}")
    print("Test set — folder 4 (Feb 24-27, unseen):")
    evaluate_test(ts_model, TEST_DIR, max_files=400)
