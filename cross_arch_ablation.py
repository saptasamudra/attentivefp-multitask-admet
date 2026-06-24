"""
Cross-Architecture Ablation: Pharma-MoE on GIN and DMPNN
"""
import json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.data import DataLoader, Data
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import ttest_rel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── TDC loader (handles both old and new API) ────────────────────────────────
def load_dataset(name):
    from tdc import single_pred

    cfg = {
        'solubility_aqsoldb': ('Solubility_AqSolDB', 'mae',   'ADME'),
        'caco2_wang':          ('Caco2_Wang',          'mae',   'ADME'),
        'ld50_zhu':            ('LD50_Zhu',             'mae',   'Tox'),
        'bbb_martins':         ('BBB_Martins',          'auroc', 'ADME'),
        'herg':                ('hERG',                 'auroc', 'Tox'),
    }
    tdc_name, metric, cls = cfg[name]
    loader_cls = getattr(single_pred, cls)
    data = loader_cls(name=tdc_name)
    split = data.get_split(method='scaffold')
    return split, metric

# ── Pharmacophore features ───────────────────────────────────────────────────
def extract_pharmacophore(smiles_list):
    feats = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            feats.append(np.zeros(7, dtype=np.float32)); continue
        feats.append(np.array([
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol) / 100.0,
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            Descriptors.MolWt(mol) / 500.0,
        ], dtype=np.float32))
    arr = np.stack(feats)
    return (arr - arr.mean(0)) / (arr.std(0) + 1e-8)

# ── PyG conversion ───────────────────────────────────────────────────────────
def smiles_to_pyg(smiles_list, labels):
    pts = []
    for smi, y in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        atoms = [[a.GetAtomicNum(), a.GetDegree(), int(a.GetIsAromatic()),
                  int(a.IsInRing()), a.GetFormalCharge(),
                  a.GetTotalNumHs(), a.GetNumRadicalElectrons()] for a in mol.GetAtoms()]
        if not atoms: continue
        x = torch.tensor(atoms, dtype=torch.float)
        ei = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ei += [[i,j],[j,i]]
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous() \
                     if ei else torch.zeros((2,0), dtype=torch.long)
        pts.append(Data(x=x, edge_index=edge_index,
                        y=torch.tensor([float(y)], dtype=torch.float)))
    return pts

# ── Encoders ─────────────────────────────────────────────────────────────────
class GINEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden, layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for i in range(layers):
            d = in_dim if i == 0 else hidden
            mlp = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(),
                                      torch.nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
            self.bns.append(torch.nn.BatchNorm1d(hidden))
    def forward(self, x, ei, batch):
        for c, bn in zip(self.convs, self.bns):
            x = F.relu(bn(c(x, ei)))
        return global_mean_pool(x, batch)

class DMPNNEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden, layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for i in range(layers):
            d = in_dim if i == 0 else hidden
            self.convs.append(GCNConv(d, hidden))
            self.bns.append(torch.nn.BatchNorm1d(hidden))
    def forward(self, x, ei, batch):
        for c, bn in zip(self.convs, self.bns):
            x = F.relu(bn(c(x, ei)))
        return global_mean_pool(x, batch)

# ── Model ─────────────────────────────────────────────────────────────────────
class PharmaModel(torch.nn.Module):
    def __init__(self, in_dim, hidden, layers, n_exp, top_k,
                 backbone, use_pharma):
        super().__init__()
        self.use_pharma = use_pharma
        self.n_exp = n_exp; self.top_k = top_k
        self.encoder = GINEncoder(in_dim, hidden, layers) if backbone == 'GIN' \
                       else DMPNNEncoder(in_dim, hidden, layers)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                torch.nn.Linear(hidden, hidden))
            for _ in range(n_exp)])
        if use_pharma:
            self.g_gate = torch.nn.Linear(hidden, n_exp)
            self.p_enc  = torch.nn.Sequential(
                torch.nn.Linear(7, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, n_exp))
            self.alpha  = torch.nn.Parameter(torch.tensor(1.0))
        else:
            self.gate = torch.nn.Linear(hidden, n_exp)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, data, pharma=None):
        h = self.encoder(data.x.float(), data.edge_index, data.batch)
        if self.use_pharma:
            a = torch.sigmoid(self.alpha)
            g = a * self.g_gate(h) + (1-a) * self.p_enc(pharma)
        else:
            g = self.gate(h)
        tv, ti = torch.topk(g, self.top_k, dim=-1)
        w = torch.zeros_like(g).scatter_(1, ti, F.softmax(tv, dim=-1))
        bal = self.n_exp * (w.mean(0)**2).sum()
        out = (w.unsqueeze(-1) * torch.stack([e(h) for e in self.experts],1)).sum(1)
        return self.head(out).squeeze(-1), bal

# ── Train + eval ─────────────────────────────────────────────────────────────
def run(tr, va, te, ph_tr, ph_va, ph_te, backbone, use_pharma, metric, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PharmaModel(tr[0].x.shape[1], 256, 3, 8, 2,
                        backbone, use_pharma).to(DEVICE)
    opt = Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    tr_loader = DataLoader(tr, batch_size=64, shuffle=True)
    best_val, best_test, wait = float('inf'), 0.0, 0

    def score_set(dataset, ph_arr):
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        preds, trues = [], []
        idx = 0
        for b in loader:
            b = b.to(DEVICE); bs = b.num_graphs
            ph = torch.tensor(ph_arr[idx:idx+bs], dtype=torch.float).to(DEVICE) \
                 if use_pharma else None
            idx += bs
            o, _ = model(b, ph)
            preds.extend(o.cpu().numpy())
            trues.extend(b.y.squeeze().cpu().numpy())
        p, t = np.array(preds), np.array(trues)
        if metric == 'mae': return float(np.mean(np.abs(p-t)))
        from sklearn.metrics import roc_auc_score
        try: return float(roc_auc_score(t, p))
        except: return 0.5

    for epoch in range(150):
        model.train()
        idx = 0
        for batch in tr_loader:
            batch = batch.to(DEVICE); bs = batch.num_graphs
            ph = torch.tensor(ph_tr[idx:idx+bs], dtype=torch.float).to(DEVICE) \
                 if use_pharma else None
            idx += bs
            out, bal = model(batch, ph)
            y = batch.y.squeeze()
            mask = ~torch.isnan(y)
            if mask.sum() == 0: continue
            loss = F.mse_loss(out[mask], y[mask]) + 0.01*bal
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vs = score_set(va, ph_va)
            improved = vs < best_val if metric=='mae' else vs > (1-best_val)
            if improved:
                best_val = vs if metric=='mae' else 1-vs
                best_test = score_set(te, ph_te)
                wait = 0
            else:
                wait += 1
                if wait >= 20: break

    return best_test

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',    nargs='+', type=int, default=[0,1,2])
    parser.add_argument('--backbones',nargs='+', default=['GIN','DMPNN'])
    parser.add_argument('--datasets', nargs='+',
        default=['solubility_aqsoldb','bbb_martins','herg','caco2_wang','ld50_zhu'])
    args = parser.parse_args()

    print('Device:', DEVICE)
    print('='*70)
    print('  Cross-Architecture Ablation: Pharma-MoE on GIN and DMPNN')
    print('='*70)

    all_results = {}
    for ds in args.datasets:
        print(f'\n{"─"*70}\n  [{ds}]')
        split, metric = load_dataset(ds)
        tr_df, va_df, te_df = split['train'], split['valid'], split['test']
        tr  = smiles_to_pyg(tr_df['Drug'].tolist(), tr_df['Y'].tolist())
        va  = smiles_to_pyg(va_df['Drug'].tolist(), va_df['Y'].tolist())
        te  = smiles_to_pyg(te_df['Drug'].tolist(), te_df['Y'].tolist())
        ph_tr = extract_pharmacophore(tr_df['Drug'].tolist())
        ph_va = extract_pharmacophore(va_df['Drug'].tolist())
        ph_te = extract_pharmacophore(te_df['Drug'].tolist())

        ds_res = {}
        for bb in args.backbones:
            ps, ss = [], []
            for seed in args.seeds:
                p = run(tr,va,te,ph_tr,ph_va,ph_te, bb, True,  metric, seed)
                s = run(tr,va,te,ph_tr,ph_va,ph_te, bb, False, metric, seed)
                ps.append(p); ss.append(s)
                g = (s-p)/s*100 if metric=='mae' else (p-s)/s*100
                print(f'  [{bb}] Seed {seed}: Pharma={p:.4f}  Standard={s:.4f}  Δ={g:+.1f}%')

            pm, sm = np.mean(ps), np.mean(ss)
            gain = (sm-pm)/sm*100 if metric=='mae' else (pm-sm)/sm*100
            comp = [-x for x in ss] if metric=='mae' else ss
            _, pval = ttest_rel(ps, comp)
            sig = '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else 'ns'
            print(f'  [{bb}] RESULT  Pharma={pm:.4f}  Standard={sm:.4f}  '
                  f'Gain={gain:+.1f}%  p={pval:.4f} {sig}')
            ds_res[bb] = {'pharma': float(pm), 'standard': float(sm),
                          'gain': float(gain), 'p': float(pval), 'sig': sig}
        all_results[ds] = ds_res

    print('\n' + '='*70)
    print('  FINAL SUMMARY')
    print('='*70)
    print(f'  {"Dataset":<28} {"GIN":>10} {"DMPNN":>10} {"Consistent":>11}')
    print('  ' + '-'*62)
    for ds, res in all_results.items():
        gg = res.get('GIN',  {}).get('gain', 0)
        dg = res.get('DMPNN',{}).get('gain', 0)
        both = '✓' if gg > 0 and dg > 0 else '✗'
        print(f'  {ds:<28} {gg:>+9.1f}% {dg:>+9.1f}% {both:>11}')

    with open('cross_arch_results.json','w') as f:
        json.dump(all_results, f, indent=2)
    print('\n[SAVED] cross_arch_results.json')

if __name__ == '__main__':
    main()
