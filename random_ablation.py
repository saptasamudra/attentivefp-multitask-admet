"""
random_ablation.py
===================
Control experiment: replace pharmacophore vectors with
random 7-dimensional Gaussian noise vectors.

If Pharma-MoE still wins vs Random-MoE → pharmacophore
features specifically matter (not just extra input dimensions).

If Random-MoE matches Pharma-MoE → improvement was from
extra parameters/dimensions, not pharmacophore semantics.

Run: python random_ablation.py
"""

import json, copy, argparse
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

PHARMA_DIM = 7

# ── Pharmacophore extractor ────────────────────────────────────────────

def extract_pharmacophore(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None: return np.zeros(PHARMA_DIM, dtype=np.float32)
        hbd  = float(rdMolDescriptors.CalcNumHBD(mol))
        hba  = float(rdMolDescriptors.CalcNumHBA(mol))
        ar   = float(rdMolDescriptors.CalcNumAromaticRings(mol))
        rot  = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        hydr = sum(1 for a in mol.GetAtoms()
                  if a.GetAtomicNum() in (6,16)
                  and not any(n.GetAtomicNum() in (7,8,9,17,35)
                             for n in a.GetNeighbors()))
        pos  = sum(1 for a in mol.GetAtoms()
                  if a.GetAtomicNum()==7 and a.GetTotalValence()<4)
        neg  = sum(1 for a in mol.GetAtoms()
                  if a.GetAtomicNum() in (8,16)
                  and any(b.GetBondTypeAsDouble()==2.0 for b in a.GetBonds()))
        return np.array([hbd,hba,ar,float(hydr),float(pos),float(neg),rot],
                       dtype=np.float32)
    except:
        return np.zeros(PHARMA_DIM, dtype=np.float32)


def extract_random(smiles):
    """Random 7-dim Gaussian noise — the control."""
    return np.random.randn(PHARMA_DIM).astype(np.float32)


def build_pharma_stats(smiles_list):
    feats = np.array([extract_pharmacophore(s) for s in smiles_list])
    return feats.mean(0), feats.std(0) + 1e-6


# ── Model ─────────────────────────────────────────────────────────────

class DualGateMoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k, aux_dim=PHARMA_DIM):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.experts     = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for _ in range(num_experts)
        ])
        self.graph_gate     = nn.Linear(in_dim, num_experts)
        self.aux_encoder    = nn.Sequential(
            nn.Linear(aux_dim, 32), nn.ReLU(),
            nn.Linear(32, num_experts))
        self.alpha = nn.Parameter(torch.tensor(0.7))

    def forward(self, x, aux_vec, return_routing=False):
        g1 = self.graph_gate(x)
        g2 = self.aux_encoder(aux_vec)
        alpha = torch.sigmoid(self.alpha)
        gate_logits = alpha * g1 + (1.0 - alpha) * g2
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.zeros_like(gate_logits).scatter_(
            1, topk_idx, F.softmax(topk_vals, dim=-1))
        load = weights.mean(0)
        bal  = self.num_experts * (load * load).sum()
        eo   = torch.stack([e(x) for e in self.experts], dim=1)
        out  = (weights.unsqueeze(-1) * eo).sum(1)
        if return_routing:
            return out, bal, weights, weights.argmax(-1)
        return out, bal


class DualGateMoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout,
                 num_experts, top_k, aux_dim=PHARMA_DIM):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = DualGateMoELayer(hidden, hidden, num_experts, top_k, aux_dim)
        self.head = nn.Linear(hidden, 1)

    def encode(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            if x.size(0) > 1: x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, b)

    def forward(self, data, aux_vec):
        m = self.encode(data)
        x, bal = self.moe(m, aux_vec)
        return self.head(x).squeeze(-1), bal


# ── Featurization ─────────────────────────────────────────────────────

def mol_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        af = []
        for atom in mol.GetAtoms():
            af.append([atom.GetAtomicNum(), int(atom.GetChiralTag()),
                       atom.GetDegree(), atom.GetFormalCharge(),
                       atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
                       int(atom.GetHybridization()),
                       int(atom.GetIsAromatic()), int(atom.IsInRing())])
        x = torch.tensor(af, dtype=torch.float)
        ei, ea = [], []
        for bond in mol.GetBonds():
            i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = [int(bond.GetBondTypeAsDouble()),
                  int(bond.GetStereo()), int(bond.GetIsConjugated())]
            ei += [[i,j],[j,i]]; ea += [bf,bf]
        if not ei:
            ei = torch.zeros((2,0), dtype=torch.long)
            ea = torch.zeros((0,3), dtype=torch.float)
        else:
            ei = torch.tensor(ei, dtype=torch.long).t().contiguous()
            ea = torch.tensor(ea, dtype=torch.float)
        return Data(x=x, edge_index=ei, edge_attr=ea)
    except: return None


def build_dataset(smiles_list, labels, pharma_mean, pharma_std,
                  use_random=False, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    dl, vs, vp = [], [], []
    for smi, lab in zip(smiles_list, labels):
        g = mol_to_graph(smi)
        if g is None: continue
        g.y = torch.tensor([float(lab)], dtype=torch.float)
        g.smiles = smi
        dl.append(g); vs.append(smi)
        if use_random:
            # Random control — Gaussian noise, same dimension
            pf = rng.randn(PHARMA_DIM).astype(np.float32)
        else:
            pf = (extract_pharmacophore(smi) - pharma_mean) / pharma_std
        vp.append(pf)
    return dl, vs, np.array(vp)


def load_tdc(tdc_name):
    for cls in ['ADME','Tox','ADMET']:
        try:
            mod = __import__('tdc.single_pred', fromlist=[cls])
            C = getattr(mod, cls)
            return C(name=tdc_name).get_split(method="scaffold", seed=42)
        except: continue
    return None


# ── Loader ────────────────────────────────────────────────────────────

class AuxLoader:
    def __init__(self, dl, pa, bs, shuffle=True, drop_last=False):
        self.dl=dl; self.pa=pa; self.bs=bs
        self.shuffle=shuffle; self.drop_last=drop_last
    def __iter__(self):
        idx = np.arange(len(self.dl))
        if self.shuffle: np.random.shuffle(idx)
        for s in range(0, len(idx), self.bs):
            bi = idx[s:s+self.bs]
            if self.drop_last and len(bi)<self.bs: continue
            bp = torch.tensor(self.pa[bi], dtype=torch.float).to(DEVICE)
            bb = Batch.from_data_list([self.dl[i] for i in bi]).to(DEVICE)
            yield bb, bp


# ── Training ──────────────────────────────────────────────────────────

def higher_is_better(metric):
    return metric in ("spearman","auroc")

def metric_fn(preds, truths, metric):
    p,t = np.array(preds), np.array(truths)
    if metric=="mae":      return float(np.mean(np.abs(p-t)))
    if metric=="spearman": r,_=spearmanr(p,t); return float(r)
    if metric=="auroc":
        from scipy.special import expit
        try: return float(roc_auc_score(t, expit(p)))
        except: return 0.5
    return 0.0

@torch.no_grad()
def evaluate(model, dl, pa, metric):
    model.eval()
    ps,ts=[],[]
    for b,p in AuxLoader(dl,pa,256,shuffle=False):
        out,_ = model(b,p)
        if out.dim()==0: out=out.unsqueeze(0)
        ps.extend(out.cpu().numpy().tolist())
        ts.extend(b.y.cpu().numpy().flatten().tolist())
    return metric_fn(ps,ts,metric)

def train_model(params, tr_dl, tr_pa, va_dl, va_pa, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    metric = params["metric"]
    hib    = higher_is_better(metric)

    model = DualGateMoEGCN(
        in_dim=9, hidden=params["hidden"],
        num_layers=params["num_layers"], dropout=params["dropout"],
        num_experts=params["num_experts"], top_k=params["top_k"],
    ).to(DEVICE)

    opt   = torch.optim.Adam(model.parameters(),
              lr=params["lr"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
              opt, patience=5, factor=0.5,
              mode="min" if not hib else "max")

    best_val=float("inf") if not hib else -float("inf")
    best_state=None; patience=0
    drop_last = len(tr_dl)%64==1

    for epoch in range(150):
        model.train()
        for b,p in AuxLoader(tr_dl,tr_pa,64,shuffle=True,drop_last=drop_last):
            opt.zero_grad()
            out,bal = model(b,p)
            y = b.y.squeeze()
            if y.dim()==0: y=y.unsqueeze(0)
            if out.dim()==0: out=out.unsqueeze(0)
            if metric in ("mae","spearman"):
                loss = F.mse_loss(out,y) + 0.01*bal
            else:
                loss = F.binary_cross_entropy_with_logits(out,y) + 0.01*bal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()

        vs = evaluate(model, va_dl, va_pa, metric)
        sched.step(vs)
        improved = (vs<best_val if not hib else vs>best_val)
        if improved: best_val=vs; best_state=copy.deepcopy(model.state_dict()); patience=0
        else:
            patience+=1
            if patience>=20: break

    model.load_state_dict(best_state)
    return model


# ── Config ────────────────────────────────────────────────────────────

BEST_PARAMS = {
    "solubility_aqsoldb":{"hidden":256,"num_layers":4,"dropout":0.153,"num_experts":8,"top_k":3,"lr":0.000896,"weight_decay":1.99e-05,"tdc_name":"Solubility_AqSolDB","metric":"mae"},
    "caco2_wang":{"hidden":256,"num_layers":3,"dropout":0.031,"num_experts":16,"top_k":4,"lr":0.000599,"weight_decay":2.01e-05,"tdc_name":"Caco2_Wang","metric":"mae"},
    "lipophilicity_astrazeneca":{"hidden":256,"num_layers":4,"dropout":0.038,"num_experts":4,"top_k":1,"lr":0.000972,"weight_decay":7.03e-05,"tdc_name":"Lipophilicity_AstraZeneca","metric":"mae"},
    "ld50_zhu":{"hidden":256,"num_layers":4,"dropout":0.099,"num_experts":16,"top_k":2,"lr":0.000470,"weight_decay":4.92e-05,"tdc_name":"LD50_Zhu","metric":"mae"},
    "vdss_lombardo":{"hidden":256,"num_layers":3,"dropout":0.222,"num_experts":8,"top_k":2,"lr":0.000905,"weight_decay":2.82e-05,"tdc_name":"VDss_Lombardo","metric":"spearman"},
    "half_life_obach":{"hidden":256,"num_layers":3,"dropout":0.220,"num_experts":16,"top_k":4,"lr":0.000539,"weight_decay":1.83e-05,"tdc_name":"Half_Life_Obach","metric":"spearman"},
    "bbb_martins":{"hidden":256,"num_layers":4,"dropout":0.202,"num_experts":16,"top_k":4,"lr":0.000527,"weight_decay":1.27e-05,"tdc_name":"BBB_Martins","metric":"auroc"},
    "herg":{"hidden":256,"num_layers":2,"dropout":0.222,"num_experts":8,"top_k":4,"lr":0.000156,"weight_decay":2.01e-05,"tdc_name":"hERG","metric":"auroc"},
    "ames":{"hidden":256,"num_layers":4,"dropout":0.191,"num_experts":16,"top_k":2,"lr":0.000316,"weight_decay":3.76e-05,"tdc_name":"AMES","metric":"auroc"},
    "dili":{"hidden":128,"num_layers":2,"dropout":0.031,"num_experts":16,"top_k":3,"lr":0.000835,"weight_decay":1.01e-05,"tdc_name":"DILI","metric":"auroc"},
}

DISPLAY = {
    "solubility_aqsoldb":"Solubility","caco2_wang":"Caco-2",
    "lipophilicity_astrazeneca":"Lipophilicity","ld50_zhu":"LD50",
    "vdss_lombardo":"VDss","half_life_obach":"Half-Life",
    "bbb_martins":"BBB","herg":"hERG","ames":"AMES","dili":"DILI",
}

SEEDS = [0,1,2,3,4]


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=list(BEST_PARAMS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    results = {}

    print("="*70)
    print("  Control Ablation: Pharmacophore vs Random 7-dim Noise")
    print("  Hypothesis: Pharma-MoE > Random-MoE proves pharmacophore")
    print("  features specifically matter, not just extra dimensions")
    print("="*70)

    for dataset in args.datasets:
        if dataset not in BEST_PARAMS: continue
        params  = BEST_PARAMS[dataset]
        metric  = params["metric"]
        display = DISPLAY.get(dataset, dataset)
        hib     = higher_is_better(metric)

        print(f"\n{'─'*70}")
        print(f"  [{display}]")

        split = load_tdc(params["tdc_name"])
        if split is None: print("  TDC failed"); continue

        pm, ps = build_pharma_stats(list(split["train"]["Drug"]))

        pharma_scores, random_scores = [], []

        for seed in args.seeds:
            # Pharmacophore model
            tr_dl,_,tr_pa = build_dataset(
                split["train"]["Drug"],split["train"]["Y"],pm,ps,
                use_random=False)
            va_dl,_,va_pa = build_dataset(
                split["valid"]["Drug"],split["valid"]["Y"],pm,ps,
                use_random=False)
            te_dl,_,te_pa = build_dataset(
                split["test"]["Drug"],split["test"]["Y"],pm,ps,
                use_random=False)

            m_pharma = train_model(params, tr_dl, tr_pa, va_dl, va_pa, seed)
            sc_pharma = evaluate(m_pharma, te_dl, te_pa, metric)
            pharma_scores.append(sc_pharma)

            # Random control model (same seed for fair comparison)
            tr_dl2,_,tr_pa2 = build_dataset(
                split["train"]["Drug"],split["train"]["Y"],pm,ps,
                use_random=True, rng_seed=seed)
            va_dl2,_,va_pa2 = build_dataset(
                split["valid"]["Drug"],split["valid"]["Y"],pm,ps,
                use_random=True, rng_seed=seed+100)
            te_dl2,_,te_pa2 = build_dataset(
                split["test"]["Drug"],split["test"]["Y"],pm,ps,
                use_random=True, rng_seed=seed+200)

            m_random = train_model(params, tr_dl2, tr_pa2, va_dl2, va_pa2, seed)
            sc_random = evaluate(m_random, te_dl2, te_pa2, metric)
            random_scores.append(sc_random)

            print(f"  Seed {seed}: Pharma={sc_pharma:.4f}  "
                  f"Random={sc_random:.4f}  "
                  f"Delta={sc_pharma-sc_random:+.4f}")

        # Stats
        pharma_arr = np.array(pharma_scores)
        random_arr = np.array(random_scores)
        mean_p = float(np.mean(pharma_arr))
        mean_r = float(np.mean(random_arr))

        if hib:
            gain = (mean_p - mean_r) / abs(mean_r) * 100
        else:
            gain = (mean_r - mean_p) / abs(mean_r) * 100

        # Paired t-test
        try:
            t_stat, p_val = ttest_rel(pharma_scores, random_scores)
        except:
            t_stat, p_val = float('nan'), 1.0

        # Wilcoxon
        try:
            w_stat, w_p = wilcoxon(pharma_scores, random_scores)
        except:
            w_stat, w_p = float('nan'), 1.0

        sig = "***" if p_val<0.001 else "**" if p_val<0.01 \
              else "*" if p_val<0.05 else "ns"

        verdict = ("✓ PHARMA SPECIFIC" if gain > 0 and p_val < 0.05
                  else "~ MARGINAL" if gain > 0
                  else "✗ NO ADVANTAGE")

        print(f"\n  ── [{display}] ──")
        print(f"  Pharma-MoE: {metric}={mean_p:.4f}±{np.std(pharma_arr):.4f}")
        print(f"  Random-MoE: {metric}={mean_r:.4f}±{np.std(random_arr):.4f}")
        print(f"  Gain vs random: {gain:+.1f}%")
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f} {sig}")
        print(f"  Wilcoxon: W={w_stat}, p={w_p:.4f}")
        print(f"  Verdict: {verdict}")

        results[dataset] = {
            "display": display, "metric": metric,
            "pharma_mean": round(mean_p, 4),
            "random_mean": round(mean_r, 4),
            "gain_vs_random_pct": round(gain, 2),
            "t_stat": round(float(t_stat), 4),
            "p_ttest": round(float(p_val), 6),
            "w_stat": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
            "p_wilcoxon": round(float(w_p), 6),
            "significant": p_val < 0.05,
            "verdict": verdict,
        }

    with open("random_ablation_results.json","w") as f:
        json.dump(results,f,indent=2)
    print(f"\n  [SAVED] random_ablation_results.json")

    print(f"\n{'='*70}")
    print("  ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Dataset':<18} {'Gain vs Random':>15} "
          f"{'p-value':>10} {'Sig':>5} Verdict")
    print("  "+"-"*65)

    sig_wins = 0
    for d,v in results.items():
        sig = ("***" if v['p_ttest']<0.001 else
               "**" if v['p_ttest']<0.01 else
               "*" if v['p_ttest']<0.05 else "ns")
        if v['significant'] and v['gain_vs_random_pct'] > 0:
            sig_wins += 1
        print(f"  {v['display']:<18} {v['gain_vs_random_pct']:>+14.1f}% "
              f"{v['p_ttest']:>10.4f} {sig:>5} {v['verdict']}")

    print(f"\n  Pharmacophore-specific wins (p<0.05): {sig_wins}/{len(results)}")
    print(f"\n  INTERPRETATION:")
    print(f"  If sig_wins > 50% of datasets → pharmacophore features")
    print(f"  specifically drive improvement, not just extra dimensions.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
