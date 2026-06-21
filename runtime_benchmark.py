# Runtime benchmark: PharmaGuidedMoE vs Standard MoE
# Run from D:/molprop_project/: python runtime_benchmark.py
# Outputs: runtime_results.json + figures/fig7_runtime.png

import time, json, os, gc
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SEEDS    = 3
N_EPOCHS   = 50
BATCH_SIZE = 64
HIDDEN     = 200
N_EXPERTS  = 4
TOP_K      = 2

print(f"Device: {DEVICE}")
print(f"Seeds: {N_SEEDS} | Epochs: {N_EPOCHS} | Batch: {BATCH_SIZE}")
print("=" * 60)

# ── Minimal MoE modules (self-contained, no external deps) ───────────────
import torch.nn as nn
import torch.nn.functional as F

class ExpertFFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d * 2, d)
        )
    def forward(self, x): return self.net(x)

class StandardMoELayer(nn.Module):
    def __init__(self, d, n_exp, k):
        super().__init__()
        self.k       = k
        self.gate    = nn.Linear(d, n_exp)
        self.experts = nn.ModuleList([ExpertFFN(d) for _ in range(n_exp)])

    def forward(self, h):
        logits  = self.gate(h)
        topk    = torch.topk(logits, self.k, dim=-1)
        weights = F.softmax(topk.values, dim=-1)
        out     = torch.zeros_like(h)
        for i, exp in enumerate(self.experts):
            mask = (topk.indices == i).any(dim=-1)
            if mask.any():
                w_i = (topk.indices == i).float() * weights
                w_i = w_i.sum(dim=-1, keepdim=True)
                out[mask] += w_i[mask] * exp(h[mask])
        return out

class PharmaGuidedMoELayer(nn.Module):
    def __init__(self, d, n_exp, k, pharma_dim=7):
        super().__init__()
        self.k            = k
        self.graph_gate   = nn.Linear(d, n_exp)
        self.pharma_gate  = nn.Linear(pharma_dim, n_exp)
        self.alpha        = nn.Parameter(torch.tensor(0.7))
        self.experts      = nn.ModuleList([ExpertFFN(d) for _ in range(n_exp)])

    def forward(self, h, pharma):
        a       = torch.sigmoid(self.alpha)
        logits  = a * self.graph_gate(h) + (1 - a) * self.pharma_gate(pharma)
        topk    = torch.topk(logits, self.k, dim=-1)
        weights = F.softmax(topk.values, dim=-1)
        out     = torch.zeros_like(h)
        for i, exp in enumerate(self.experts):
            mask = (topk.indices == i).any(dim=-1)
            if mask.any():
                w_i = (topk.indices == i).float() * weights
                w_i = w_i.sum(dim=-1, keepdim=True)
                out[mask] += w_i[mask] * exp(h[mask])
        return out

class StandardModel(nn.Module):
    def __init__(self, in_d, hidden, n_exp, k):
        super().__init__()
        self.enc  = nn.Sequential(nn.Linear(in_d, hidden), nn.ReLU())
        self.moe  = StandardMoELayer(hidden, n_exp, k)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, _pharma=None):
        h = self.enc(x)
        h = self.moe(h)
        return self.head(h).squeeze(-1)

class PharmaModel(nn.Module):
    def __init__(self, in_d, hidden, n_exp, k):
        super().__init__()
        self.enc  = nn.Sequential(nn.Linear(in_d, hidden), nn.ReLU())
        self.moe  = PharmaGuidedMoELayer(hidden, n_exp, k)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, pharma):
        h = self.enc(x)
        h = self.moe(h, pharma)
        return self.head(h).squeeze(-1)

# ── Synthetic dataset (matches real molecule feature dims) ───────────────
def make_dataset(n, in_d=39, pharma_d=7):
    X      = torch.randn(n, in_d)
    pharma = torch.randn(n, pharma_d)
    y      = torch.randn(n)
    return X, pharma, y

def time_model(ModelClass, dataset_size, name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, pharma, y = make_dataset(dataset_size)
    X      = X.to(DEVICE)
    pharma = pharma.to(DEVICE)
    y      = y.to(DEVICE)

    model = ModelClass(39, HIDDEN, N_EXPERTS, TOP_K).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # warmup 2 epochs
    for _ in range(2):
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i+BATCH_SIZE]
            pb = pharma[i:i+BATCH_SIZE]
            yb = y[i:i+BATCH_SIZE]
            opt.zero_grad()
            pred = model(xb, pb)
            loss_fn(pred, yb).backward()
            opt.step()

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_EPOCHS):
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i+BATCH_SIZE]
            pb = pharma[i:i+BATCH_SIZE]
            yb = y[i:i+BATCH_SIZE]
            opt.zero_grad()
            pred = model(xb, pb)
            loss_fn(pred, yb).backward()
            opt.step()

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0

    # param count
    params = sum(p.numel() for p in model.parameters())

    del model, X, pharma, y
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return elapsed, params

# ── Benchmark configs ────────────────────────────────────────────────────
configs = [
    ('Solubility', 9980),
    ('hERG',       655),
]

results = {}

for ds_name, ds_size in configs:
    print(f"\n{'─'*60}")
    print(f"  [{ds_name}]  n={ds_size}")
    results[ds_name] = {}

    for label, ModelCls in [('Standard MoE', StandardModel),
                             ('PharmaGuidedMoE', PharmaModel)]:
        times = []
        for seed in range(N_SEEDS):
            t, params = time_model(ModelCls, ds_size, label, seed)
            times.append(t)
            print(f"    {label} seed {seed}: {t:.2f}s")

        mean_t = float(np.mean(times))
        std_t  = float(np.std(times))
        results[ds_name][label] = {
            'mean_sec':  round(mean_t, 3),
            'std_sec':   round(std_t, 3),
            'n_params':  params,
            'per_epoch_ms': round(mean_t / N_EPOCHS * 1000, 2),
        }
        print(f"    → {label}: {mean_t:.2f}±{std_t:.2f}s "
              f"({mean_t/N_EPOCHS*1000:.1f} ms/epoch) | params={params:,}")

# overhead ratio
for ds in results:
    std_t   = results[ds]['Standard MoE']['mean_sec']
    pharma_t= results[ds]['PharmaGuidedMoE']['mean_sec']
    overhead = (pharma_t - std_t) / std_t * 100
    results[ds]['overhead_pct'] = round(overhead, 2)
    print(f"\n  [{ds}] overhead: {overhead:+.1f}%")

with open('runtime_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✓ runtime_results.json saved")

# ── Figure 7 ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BLUE  = '#2563EB'
AMBER = '#D97706'
PANEL = '#F8FAFC'

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), facecolor='white')
fig.subplots_adjust(wspace=0.4)

ds_names = list(results.keys())

for ax_idx, ds in enumerate(ds_names):
    ax  = axes[ax_idx]
    ax.set_facecolor(PANEL)

    models   = ['Standard MoE', 'PharmaGuidedMoE']
    means    = [results[ds][m]['mean_sec'] for m in models]
    stds     = [results[ds][m]['std_sec']  for m in models]
    colors   = [BLUE, AMBER]
    labels   = ['Standard\nMoE', 'Pharma-\nGuidedMoE']
    overhead = results[ds]['overhead_pct']

    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.85,
                  width=0.5, capsize=4, error_kw={'linewidth':1},
                  edgecolor='white', linewidth=0.5)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                m + s + max(means)*0.02,
                f'{m:.1f}s', ha='center', va='bottom', fontsize=8.5,
                fontweight='bold')

    ax.annotate(f'overhead:\n{overhead:+.1f}%',
                xy=(1, means[1]), xytext=(1.25, (means[0]+means[1])/2),
                fontsize=8, color='#B45309', style='italic',
                arrowprops=dict(arrowstyle='->', color='#B45309', lw=0.8))

    n_std   = results[ds]['Standard MoE']['n_params']
    n_ph    = results[ds]['PharmaGuidedMoE']['n_params']
    param_oh= (n_ph - n_std) / n_std * 100

    ds_sizes = {"Solubility": 9980, "hERG": 655}
    ax.set_title(f'{ds}  (n={ds_sizes[ds]:,})\n'
                 f'Params: {n_std:,} -> {n_ph:,} ({param_oh:+.1f}%)',
                 fontsize=8.5, fontweight='bold', pad=6)
    ax.set_ylabel(f'Wall-clock time ({N_EPOCHS} epochs, {N_SEEDS} seeds avg)',
                  fontsize=8.5)
    ax.tick_params(axis='both', labelsize=8.5)
    ax.set_ylim(0, max(means) * 1.35)

fig.text(0.5, -0.04,
         f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}. '
         f'Batch size {BATCH_SIZE}, hidden dim {HIDDEN}, {N_EXPERTS} experts top-{TOP_K}. '
         'Overhead from dual-gate routing and 7D pharmacophore encoder forward pass.',
         ha='center', fontsize=7, color='#64748B', style='italic')

fig.savefig('figures/fig7_runtime.png', facecolor='white')
plt.close()
print("✓ figures/fig7_runtime.png saved")
print("\nDone. Copy figures/fig7_runtime.png to paper figures folder.")
