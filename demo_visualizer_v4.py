# demo_visualizer_v4.py
# ─────────────────────────────────────────────────────────────────────────────
# Visual demo using the calibrated v4 TemporalVehicleTransformer
# Fixes vs original:
#   1. Uses best_model_v4_calibrated.pt (not v1 PatchTransformer)
#   2. Filters to BIN Time files only
#   3. Parses filename timestamps → time-of-day features fed to model
#   4. Fixed signal panel autoscale (percentile clip, not raw ylim)
#   5. Real model probability output (not step function)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert, medfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import re
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE       = Path("/home/augustin/Documents/IA_project/Spintronic Sensor Dataset for Vehicle Detection and Car Model Recognition (14 Days, Unlabeled with Timestamps)")
TEST_DIR   = BASE / "Run 20250224 14h55m20s - 20250227 14h38m27s"
MODEL_PATH = BASE / "transformer_output_v4" / "best_model_v4_calibrated.pt"
OUTPUT_DIR = BASE / "demo_output_v4"

SAMPLE_RATE  = 2400; CARRIER_HZ = 24.9; CARRIER_BW = 5.0
EDGE_TRIM    = int(0.5 * SAMPLE_RATE)
DETREND_K    = int(5.0 * SAMPLE_RATE)
DETREND_K    = DETREND_K if DETREND_K % 2 == 1 else DETREND_K + 1
IQR_CLIP     = 0.5
SEQ_LEN      = 2400; STEP_SIZE = 240; CONTEXT_WINS = 16
N_FILES_DEMO = 150    # ~2.9 hours

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── SIGNAL PROCESSING ─────────────────────────────────────────────────────────
_bp_lo = (CARRIER_HZ - CARRIER_BW) / (SAMPLE_RATE / 2)
_bp_hi = (CARRIER_HZ + CARRIER_BW) / (SAMPLE_RATE / 2)
BP_B, BP_A = butter(4, [_bp_lo, _bp_hi], btype='band')
TIMESTAMP_RE = re.compile(r"Start@ (\d{8}) - (\d{2})h(\d{2})m(\d{2})s")

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

def parse_timestamp(fname):
    m = TIMESTAMP_RE.search(str(fname))
    if not m: return None
    return datetime.strptime(
        f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}", "%Y%m%d%H%M%S")

def time_features(dt):
    if dt is None: return np.zeros(6, dtype=np.float32)
    h = (dt.hour + dt.minute/60.0) / 24.0
    d = dt.weekday() / 7.0
    return np.array([np.sin(2*np.pi*h), np.cos(2*np.pi*h),
                     np.sin(2*np.pi*d), np.cos(2*np.pi*d),
                     h, float(7<=dt.hour<=19)], dtype=np.float32)

# ── MODEL (TemporalVehicleTransformer — must match v4 exactly) ────────────────
PATCH_SIZE=50; D_MODEL=128; N_HEADS=4; N_LAYERS=4; D_FF=256; DROPOUT=0.0

class PatchEmbed1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(PATCH_SIZE, D_MODEL)
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
            nn.Dropout(DROPOUT), nn.Linear(D_MODEL//2, 2))

    def encode_window(self, x):
        B = x.shape[0]
        p = self.patch_embed(x) + self.patch_pos
        cls = self.patch_cls.expand(B,-1,-1)
        return self.patch_norm(self.patch_encoder(torch.cat([cls,p],1))[:,0])

    def forward(self, x, t):
        B  = x.shape[0]
        wv = self.encode_window(x.reshape(B*CONTEXT_WINS,SEQ_LEN,1))
        wv = wv.reshape(B, CONTEXT_WINS, D_MODEL)
        ctx = torch.cat([self.time_proj(t).unsqueeze(1), wv], 1) + self.context_pos
        cls = self.context_cls.expand(B,-1,-1)
        out = self.context_norm(self.context_encoder(torch.cat([cls,ctx],1))[:,0])
        return self.classifier(out)

class TemperatureScaler(nn.Module):
    def __init__(self, model, T):
        super().__init__()
        self.model = model
        self.T = T
    def forward(self, x, t):
        return self.model(x, t) / self.T

# ── INFERENCE ON ONE FILE ─────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, normed, tf):
    n_wins = max(0, (len(normed) - SEQ_LEN) // STEP_SIZE + 1)
    if n_wins == 0:
        return np.array([])

    wins = np.lib.stride_tricks.as_strided(
        normed, shape=(n_wins, SEQ_LEN),
        strides=(normed.strides[0]*STEP_SIZE, normed.strides[0])
    ).copy()

    probs = np.zeros(n_wins, dtype=np.float32)
    buf   = np.zeros((CONTEXT_WINS, SEQ_LEN), dtype=np.float32)
    tf_t  = torch.from_numpy(tf[np.newaxis]).to(DEVICE)

    # Batch inference with context buffer
    BSIZE = 256
    batch_x, batch_i = [], []

    for i in range(n_wins):
        buf = np.roll(buf, -1, axis=0)
        buf[-1] = wins[i]
        if i < CONTEXT_WINS - 1:
            continue
        batch_x.append(buf.copy())
        batch_i.append(i)

        if len(batch_x) == BSIZE or i == n_wins - 1:
            x_t = torch.from_numpy(
                np.stack(batch_x)[:,:,:,np.newaxis]).to(DEVICE)
            tf_b = tf_t.expand(len(batch_x), -1)
            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE=="cuda")):
                logits = model(x_t, tf_b)
            p = F.softmax(logits, dim=1)[:,1].cpu().numpy()
            for j, idx in enumerate(batch_i):
                probs[idx] = p[j]
            batch_x, batch_i = [], []

    return probs

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load v4 calibrated model
    print(f"Loading model: {MODEL_PATH}")
    base_model = TemporalVehicleTransformer()
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    base_model.load_state_dict(ckpt["model_state"])
    T_val = ckpt.get("temperature", 1.0)
    model = TemperatureScaler(base_model, T_val).to(DEVICE)
    model.eval()
    print(f"  Loaded — val F1={ckpt.get('f1','?'):.1%}  T={T_val:.4f}")

    # BIN Time files only
    test_files = sorted(
        (f for f in TEST_DIR.rglob("*.bsd") if "BIN Time" in f.name),
        key=lambda f: f.name)[:N_FILES_DEMO]
    print(f"Processing {len(test_files)} BIN Time files...")

    all_signal = []; all_probs = []; all_times = []
    t_offset = 0.0

    for i, bsd in enumerate(test_files):
        try:
            raw    = np.frombuffer(bsd.read_bytes(), dtype=np.int16)
            normed = detrend_normalize(demodulate(raw))
            dt     = parse_timestamp(bsd.name)
            tf     = time_features(dt)
            probs  = run_inference(model, normed, tf)
        except Exception as e:
            print(f"  Error {bsd.name}: {e}")
            continue

        n   = len(normed)
        n_w = len(probs)
        t_w = t_offset + (np.arange(n_w)*STEP_SIZE + SEQ_LEN//2) / SAMPLE_RATE

        all_signal.append(normed)
        all_probs.append(probs)
        all_times.append(t_w)
        t_offset += n / SAMPLE_RATE

        if i % 20 == 0:
            n_veh = int((probs > 0.5).sum())
            ts = f"{dt.strftime('%H:%M') if dt else '??:??'}"
            print(f"  [{100*(i+1)/len(test_files):5.1f}%]  "
                  f"file {i+1}  {ts}  detections: {n_veh}/{n_w}")

    signal_cat = np.concatenate(all_signal)
    probs_cat  = np.concatenate(all_probs)
    times_cat  = np.concatenate(all_times)

    print(f"\nDuration : {t_offset/60:.1f} min")
    print(f"Windows  : {len(probs_cat):,}")
    print(f"Detections: {int((probs_cat>0.5).sum()):,} "
          f"({100*(probs_cat>0.5).mean():.2f}%)")

    # ── Find events ───────────────────────────────────────────────────────────
    detected = probs_cat > 0.5
    events   = []
    in_ev = False; t_start = 0; max_conf = 0
    for i, (t, d, p) in enumerate(zip(times_cat, detected, probs_cat)):
        if d and not in_ev:
            in_ev = True; t_start = t; max_conf = p
        elif d and in_ev:
            max_conf = max(max_conf, p)
        elif not d and in_ev:
            if t - t_start > 0.3:   # min 0.3s event
                events.append((t_start, t, float(max_conf)))
            in_ev = False
    if in_ev:
        events.append((t_start, times_cat[-1], float(max_conf)))

    print(f"Events   : {len(events)}")

    # ── Plot 1: Full overview ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 10), facecolor="#0a0a0f")
    gs  = GridSpec(3, 1, figure=fig, hspace=0.08,
                   height_ratios=[2, 1.5, 0.8],
                   top=0.93, bottom=0.07, left=0.05, right=0.96)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    t_min = times_cat / 60

    # Panel 1: 30s RMS energy — readable at any duration
    win_rms  = int(30 * SAMPLE_RATE)   # 30s window
    n_rms    = len(signal_cat) // win_rms
    rms_vals = np.array([
        np.sqrt(np.mean(signal_cat[i*win_rms:(i+1)*win_rms]**2))
        for i in range(n_rms)], dtype=np.float32)
    rms_time = (np.arange(n_rms) + 0.5) * win_rms / SAMPLE_RATE / 60

    ax1.set_facecolor("#0d0d1a")
    ax1.fill_between(rms_time, rms_vals, alpha=0.4, color="#3a9fd8")
    ax1.plot(rms_time, rms_vals, lw=1.0, color="#3a9fd8")
    ax1.axhline(1.5, color="#ffcc44", lw=0.8, ls="--", alpha=0.7,
                label="1.5σ threshold")
    # Mark events on signal panel
    for t_s, t_e, conf in events:
        ax1.axvspan(t_s/60, t_e/60, color="#00ff88", alpha=0.25)
    ax1.set_ylim(0, max(float(np.percentile(rms_vals, 99))*1.5, 2.0))
    ax1.set_ylabel("Signal (σ)", color="#aaaacc", fontsize=10)
    ax1.set_xlim(0, t_offset/60)
    ax1.tick_params(colors="#555577", labelbottom=False)
    ax1.spines[:].set_color("#222233")
    ax1.set_title(
        f"Vehicle Detection Demo — INL Gate Sensor  "
        f"({len(events)} vehicles detected in {t_offset/60:.0f} min)"
        f"  |  Model: TemporalTransformer v4  |  T={T_val:.3f}",
        color="white", fontsize=12, fontweight="bold", pad=8)
    ax1.legend(loc="upper right", facecolor="#0d0d1a",
               edgecolor="#333344", labelcolor="white", fontsize=8)

    # Probability panel — color by value
    ax2.set_facecolor("#0d0d1a")
    ax2.fill_between(t_min, probs_cat,
                     where=probs_cat > 0.5, color="#00ff88", alpha=0.5,
                     label="Vehicle detected")
    ax2.fill_between(t_min, probs_cat,
                     where=probs_cat <= 0.5, color="#3a9fd8", alpha=0.3,
                     label="No vehicle")
    ax2.plot(t_min, probs_cat, lw=0.3, color="#3a9fd8", alpha=0.6)
    ax2.axhline(0.5, color="#ff6644", lw=1.0, ls="--", alpha=0.8,
                label="Decision threshold")
    ax2.set_ylabel("P(vehicle)", color="#aaaacc", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, t_offset/60)
    ax2.tick_params(colors="#555577", labelbottom=False)
    ax2.spines[:].set_color("#222233")
    ax2.legend(loc="upper right", facecolor="#0d0d1a",
               edgecolor="#333344", labelcolor="white", fontsize=8)

    # Event timeline
    ax3.set_facecolor("#0d0d1a")
    if events:
        ev_t    = np.array([(t_s+t_e)/2/60 for t_s,t_e,_ in events])
        ev_conf = np.array([c for _,_,c in events])
        ev_cols = [plt.cm.RdYlGn(c) for c in ev_conf]
        for xi, yi, col in zip(ev_t, ev_conf, ev_cols):
            ax3.vlines(xi, 0, yi, color=col, linewidth=1.5, alpha=0.8)
            ax3.plot(xi, yi, "o", color=col, markersize=4)
    ax3.set_xlim(0, t_offset/60)
    ax3.set_ylim(0, 1.1)
    ax3.set_xlabel("Time (minutes)", color="#aaaacc", fontsize=10)
    ax3.set_ylabel("Events", color="#aaaacc", fontsize=10)
    ax3.tick_params(colors="#555577"); ax3.set_yticks([])
    ax3.spines[:].set_color("#222233")
    sm  = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0,1))
    cax = fig.add_axes([0.965, 0.07, 0.007, 0.18])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Confidence", color="#aaaacc", fontsize=8)
    cb.ax.tick_params(colors="#555577", labelsize=7)

    out1 = OUTPUT_DIR / "demo_full_overview.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.close()
    print(f"\nSaved → {out1}")

    # ── Plot 2: Zoomed events ─────────────────────────────────────────────────
    n_show = min(9, len(events))
    if n_show > 0:
        fig2, axes = plt.subplots(3, 3, figsize=(18, 10), facecolor="#0a0a0f")
        fig2.suptitle("Individual Vehicle Events — Zoomed  (Real Model Probability)",
                      color="white", fontsize=13, fontweight="bold", y=0.98)
        sig_times = np.arange(len(signal_cat)) / SAMPLE_RATE

        for idx in range(9):
            ax = axes[idx//3][idx%3]
            ax.set_facecolor("#0d0d1a")
            ax.spines[:].set_color("#222233")
            ax.tick_params(colors="#555577")
            if idx >= n_show:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        color="#333344", fontsize=20, transform=ax.transAxes)
                continue

            t_s, t_e, conf = events[idx]
            pad = 3.0
            lo  = max(0, t_s - pad); hi = min(sig_times[-1], t_e + pad)
            ms  = (sig_times >= lo) & (sig_times <= hi)
            mw  = (times_cat >= lo) & (times_cat <= hi)
            t_p = sig_times[ms] - lo; s_p = signal_cat[ms]
            t_w = times_cat[mw] - lo; p_w = probs_cat[mw]

            ax.plot(t_p, s_p, lw=0.6, color="#3a9fd8", alpha=0.9)
            ax.axvspan(t_s-lo, t_e-lo, color="#00ff88", alpha=0.15)
            ax.axhline(1.5, color="#ffcc44", lw=0.6, ls="--", alpha=0.5)

            ax2b = ax.twinx()
            ax2b.fill_between(t_w, p_w, alpha=0.3, color="#00ff88")
            ax2b.plot(t_w, p_w, lw=1.8, color="#00ff88")
            ax2b.axhline(0.5, color="#ff6644", lw=0.8, ls="--")
            ax2b.set_ylim(-0.05, 1.3)
            ax2b.set_ylabel("P(vehicle)", color="#00ff88", fontsize=7)
            ax2b.tick_params(colors="#555577", labelsize=6)

            # Signal autoscale per event
            s_abs = np.abs(s_p)
            ymax  = max(float(np.percentile(s_abs, 99)) * 1.5, 2.5)
            ax.set_ylim(-ymax, ymax)
            ax.set_title(
                f"Event {idx+1}  t={t_s/60:.2f}min  "
                f"conf={conf:.0%}  dur={t_e-t_s:.1f}s",
                color="#aaaacc", fontsize=8, pad=4)
            ax.set_xlabel("Time (s)", color="#555577", fontsize=7)
            ax.set_ylabel("Signal (σ)", color="#555577", fontsize=7)

        plt.tight_layout()
        out2 = OUTPUT_DIR / "demo_vehicle_events.png"
        fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
        plt.close()
        print(f"Saved → {out2}")

    # ── HTML report ───────────────────────────────────────────────────────────
    event_rows = ""
    for i, (t_s, t_e, conf) in enumerate(events, 1):
        mins  = int(t_s // 60); secs = t_s % 60; dur = t_e - t_s
        color = "#00ff88" if conf>0.8 else "#ffcc44" if conf>0.6 else "#ff6644"
        event_rows += f"""
        <tr>
          <td>{i}</td><td>{mins:02d}:{secs:05.2f}</td>
          <td>{dur:.2f}s</td>
          <td style="color:{color};font-weight:bold">{conf:.1%}</td>
        </tr>"""

    mean_conf = np.mean([c for _,_,c in events]) if events else 0
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Vehicle Detection Report — INL Gate v4</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#08080f;color:#c8c8e8;font-family:'Inter',sans-serif;padding:40px}}
  h1{{font-family:'Space Mono',monospace;font-size:1.8rem;color:#00ff88;margin-bottom:4px}}
  .sub{{color:#555577;font-size:0.8rem;font-family:'Space Mono',monospace;margin-bottom:32px}}
  .badge{{display:inline-block;background:#0d2d1a;color:#00ff88;border:1px solid #00ff8844;
          border-radius:4px;padding:2px 8px;font-size:0.75rem;font-family:'Space Mono',monospace;margin-left:8px}}
  .stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:40px}}
  .stat{{background:#0d0d1a;border:1px solid #1a1a2e;border-radius:8px;padding:20px}}
  .stat-val{{font-family:'Space Mono',monospace;font-size:2rem;color:#00ff88;font-weight:700}}
  .stat-label{{color:#555577;font-size:0.72rem;margin-top:4px;text-transform:uppercase;letter-spacing:1px}}
  .section{{font-family:'Space Mono',monospace;color:#3a9fd8;font-size:0.85rem;
            text-transform:uppercase;letter-spacing:2px;margin:36px 0 14px}}
  img{{width:100%;border-radius:8px;border:1px solid #1a1a2e;margin-bottom:16px}}
  table{{width:100%;border-collapse:collapse;font-family:'Space Mono',monospace;font-size:0.78rem}}
  th{{background:#0d0d1a;color:#555577;padding:10px 16px;text-align:left;
      text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #1a1a2e}}
  td{{padding:8px 16px;border-bottom:1px solid #111122;color:#aaaacc}}
  tr:hover td{{background:#0d0d1a}}
</style></head><body>
<h1>Vehicle Detection Report <span class="badge">v4 Calibrated</span></h1>
<div class="sub">INL Gate Spintronic Sensor · Test Set: Feb 24–27 2025 · TemporalTransformer v4 · T={T_val:.4f}</div>
<div class="stats">
  <div class="stat"><div class="stat-val">{len(events)}</div><div class="stat-label">Vehicles Detected</div></div>
  <div class="stat"><div class="stat-val">{t_offset/60:.0f}m</div><div class="stat-label">Duration Analysed</div></div>
  <div class="stat"><div class="stat-val">{100*(probs_cat>0.5).mean():.2f}%</div><div class="stat-label">Vehicle Window %</div></div>
  <div class="stat"><div class="stat-val">{mean_conf:.0%}</div><div class="stat-label">Mean Confidence</div></div>
</div>
<div class="section">Signal Overview</div>
<img src="demo_full_overview.png" alt="Overview">
<img src="demo_vehicle_events.png" alt="Events">
<div class="section">Event Log ({len(events)} events)</div>
<table><thead><tr><th>#</th><th>Time</th><th>Duration</th><th>Confidence</th></tr></thead>
<tbody>{event_rows}</tbody></table>
</body></html>"""

    out3 = OUTPUT_DIR / "demo_report.html"
    out3.write_text(html)
    print(f"Saved → {out3}")
    print(f"\n✓ Demo v4 complete! Open: {out3}")

if __name__ == "__main__":
    main()
