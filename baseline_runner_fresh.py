import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GINConv
import numpy as np
from sklearn.metrics import mean_squared_error
from rdkit import Chem
import warnings

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization())
            ])
        
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])
        
        if not node_features or not edge_indices:
            return None
        
        return (
            torch.tensor(node_features, dtype=torch.float32),
            torch.tensor(edge_indices, dtype=torch.long).T
        )
    except:
        return None

class GINModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_in = nn.Linear(4, 32)
        self.conv1 = GINConv(nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)))
        self.lin_out = nn.Linear(32, 1)
    
    def forward(self, x, edge_index, batch):
        x = torch.relu(self.lin_in(x))
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        
        batch_size = batch.max().item() + 1
        x_pooled = torch.zeros(batch_size, 32, device=x.device)
        x_pooled.scatter_add_(0, batch.unsqueeze(1).expand(-1, 32), x)
        x_pooled = x_pooled / (batch.bincount().unsqueeze(1) + 1e-8)
        
        return self.lin_out(x_pooled).squeeze(-1)

def load_mock_data(name):
    if name == "ESOL":
        smiles = ['CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'CC(=O)Nc1ccc(O)cc1', 'c1ccccc1'] * 50
        targets = np.random.randn(150)
    elif name == "FreeSolv":
        smiles = ['CC(C)CC(N)C(=O)O', 'c1cc(O)ccc1'] * 50
        targets = np.random.randn(100)
    elif name == "Lipophilicity":
        smiles = ['CC(C)Cc1ccc(cc1)C(C)C', 'c1ccccc1N'] * 50
        targets = np.random.randn(100)
    else:
        smiles = ['c1ccccc1'] * 50
        targets = np.random.randn(50)
    return smiles, targets

def train_baseline(name, smiles_list, targets):
    print(f"\n{name:<15}", end=" | ")
    
    graphs = []
    for s, t in zip(smiles_list, targets):
        g = smiles_to_graph(s)
        if g:
            x, edge_index = g
            graphs.append({'x': x, 'edge_index': edge_index, 'y': torch.tensor(t, dtype=torch.float32)})
    
    if len(graphs) < 10:
        print("SKIP (insufficient data)")
        return None
    
    n_train = int(0.8 * len(graphs))
    train_g = graphs[:n_train]
    test_g = graphs[n_train:]
    
    model = GINModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(5):
        model.train()
        for g in train_g:
            x = g['x'].to(DEVICE)
            edge_index = g['edge_index'].to(DEVICE)
            y = g['y'].to(DEVICE)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
            
            optimizer.zero_grad()
            pred = model(x, edge_index, batch)
            loss = criterion(pred.unsqueeze(0), y.unsqueeze(0))
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds, targs = [], []
        for g in test_g:
            x = g['x'].to(DEVICE)
            edge_index = g['edge_index'].to(DEVICE)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
            pred = model(x, edge_index, batch).cpu().numpy()
            preds.append(pred)
            targs.append(g['y'].numpy())
        
        rmse = np.sqrt(mean_squared_error(targs, preds))
        print(f"RMSE={rmse:.3f}")
        return rmse

datasets = ["ESOL", "FreeSolv", "Lipophilicity", "PCBA", "MUV", "HIV", "BBBP", "Tox21"]

print("="*70)
for ds in datasets:
    smiles, targets = load_mock_data(ds)
    train_baseline(ds, smiles, targets)
print("="*70)