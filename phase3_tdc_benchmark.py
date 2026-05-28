"""
phase3_tdc_benchmark.py — TDC ADMET Benchmark Group
Evaluates MoE-GCN on the TDC ADMET benchmark group (22 datasets).
Uses the official admet_group API with standardized scaffold splits.

Run: python phase3_tdc_benchmark.py
Results: results_tdc.json
"""

import os, json, time, warnings, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS    = 3
EPOCHS     = 80
PATIENCE   = 15
BATCH_SIZE = 64
SAVE_PATH  = "results_tdc.json"

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── MoE-GCN ──────────────────────────────────────────────────────────────────
class MoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts; self.top_k = top_k
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()) for _ in range(num_experts)])
        self.gate = nn.Linear(in_dim, num_experts)
    def forward(self, x):
        gl = self.gate(x); tv, ti = torch.topk(gl, self.top_k, dim=-1)
        w = torch.zeros_like(gl).scatter_(1, ti, F.softmax(tv, dim=-1))
        bal = self.num_experts * (w.mean(0)**2).sum()
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        return (w.unsqueeze(-1)*eo).sum(dim=1), bal

class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden=256, num_layers=4, dropout=0.1, num_experts=4, top_k=2, num_tasks=1):
        super().__init__()
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList(); self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, num_tasks)
    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei))); x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, b); x, bal = self.moe(x)
        return self.head(x), bal

# ── SMILES → graph ────────────────────────────────────────────────────────────
def smiles_to_graph(smiles, label):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return None
    feats = []
    for atom in mol.GetAtoms():
        feats.append([atom.GetAtomicNum(), atom.GetChiralTag().real, atom.GetDegree(),
                      atom.GetFormalCharge(), atom.GetNumExplicitHs(),
                      atom.GetNumRadicalElectrons(), atom.GetHybridization().real,
                      int(atom.GetIsAromatic()), int(atom.IsInRing())])
    if not feats: return None
    x = torch.tensor(feats, dtype=torch.float)
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i,j]; dst += [j,i]
    ei = torch.tensor([src,dst], dtype=torch.long) if src else torch.zeros((2,0), dtype=torch.long)
    return Data(x=x, edge_index=ei, y=torch.tensor([[label]], dtype=torch.float))

def df_to_graphs(df):
    graphs = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row['Drug'], row['Y'])
        if g is not None: graphs.append(g)
    return graphs

# ── Eval ──────────────────────────────────────────────────────────────────────
def evaluate(model, loader, metric):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE); out, _ = model(batch)
            preds.extend(out.squeeze().cpu().numpy().flatten())
            labels.extend(batch.y.squeeze().cpu().numpy().flatten())
    preds, labels = np.array(preds), np.array(labels)
    mask = ~np.isnan(labels); preds, labels = preds[mask], labels[mask]
    if metric == "AUROC":
        if len(np.unique(labels)) < 2: return 0.0
        return float(roc_auc_score(labels, torch.sigmoid(torch.tensor(preds)).numpy()))
    elif metric == "MAE":
        return float(np.mean(np.abs(preds - labels)))
    elif metric == "RMSE":
        return float(np.sqrt(np.mean((preds-labels)**2)))
    elif metric == "Spearman":
        from scipy.stats import spearmanr
        return float(spearmanr(preds, labels).correlation)
    elif metric == "PCC":
        return float(np.corrcoef(preds, labels)[0,1])
    return 0.0

# ── Train ─────────────────────────────────────────────────────────────────────
def train_eval(train_g, val_g, test_g, in_dim, metric):
    is_classif = metric == "AUROC"
    higher_better = metric in ["AUROC", "Spearman", "PCC"]
    seed_scores = []
    vl = DataLoader(val_g,  batch_size=BATCH_SIZE)
    tl = DataLoader(test_g, batch_size=BATCH_SIZE)

    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        model = MoEGCN(in_dim).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        best_val = 0.0 if higher_better else float("inf")
        pat, best_state = 0, copy.deepcopy(model.state_dict())

        for _ in range(EPOCHS):
            model.train()
            for batch in DataLoader(train_g, batch_size=BATCH_SIZE, shuffle=True):
                batch = batch.to(DEVICE); opt.zero_grad()
                out, bal = model(batch); y = batch.y.float().squeeze()
                loss = (F.binary_cross_entropy_with_logits(out.squeeze(), y) if is_classif
                        else F.mse_loss(out.squeeze(), y)) + 0.01*bal
                loss.backward(); opt.step()

            val_s = evaluate(model, vl, metric)
            sched.step(-val_s if higher_better else val_s)
            improved = val_s > best_val if higher_better else val_s < best_val
            if improved: best_val=val_s; best_state=copy.deepcopy(model.state_dict()); pat=0
            else: pat+=1
            if pat >= PATIENCE: break

        model.load_state_dict(best_state)
        score = evaluate(model, tl, metric)
        seed_scores.append(score)
        print(f"    Seed {seed} → {metric}: {score:.4f}")

    return float(np.mean(seed_scores)), float(np.std(seed_scores))

# ── Main ─────────────────────────────────────────────────────────────────────
from tdc.benchmark_group import admet_group

results = {}
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH) as f: results = json.load(f)
    print(f"Resuming — {len(results)} done: {list(results.keys())}")

group = admet_group(path='tdc_data/')
benchmark_names = group.dataset_names
print(f"\nTDC ADMET datasets: {len(benchmark_names)}")
print(benchmark_names)

for name in benchmark_names:
    if name in results:
        print(f"  Skipping {name}")
        continue

    print(f"\n{'='*55}")
    print(f"  TDC | {name}")
    print(f"{'='*55}")
    t0 = time.time()

    try:
        benchmark = group.get(name)
        train_df  = benchmark['train_val']
        test_df   = benchmark['test']
        metric    = benchmark.get('metric', 'AUROC')

        # Split train into train/val
        val_size = max(int(len(train_df) * 0.1), 10)
        val_df   = train_df.sample(n=val_size, random_state=42)
        train_df2 = train_df.drop(val_df.index)

        print(f"  train={len(train_df2)}, val={len(val_df)}, test={len(test_df)}, metric={metric}")

        train_g = df_to_graphs(train_df2)
        val_g   = df_to_graphs(val_df)
        test_g  = df_to_graphs(test_df)

        if not train_g or not test_g:
            print(f"  Skipped — empty graphs")
            continue

        in_dim = train_g[0].x.shape[1]
        mean_s, std_s = train_eval(train_g, val_g, test_g, in_dim, metric)
        elapsed = time.time() - t0

        results[name] = {"mean": mean_s, "std": std_s, "metric": metric, "time_min": round(elapsed/60,1)}
        with open(SAVE_PATH, "w") as f: json.dump(results, f, indent=2)
        print(f"  ✓ {name}: {mean_s:.4f} ± {std_s:.4f}  ({elapsed/60:.1f} min)")

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print(f"\n{'='*55}\n  TDC ADMET SUMMARY\n{'='*55}")
for name, r in results.items():
    print(f"  {name:40} {r['metric']:8}: {r['mean']:.4f} ± {r['std']:.4f}")
print(f"\nSaved → {SAVE_PATH}")
