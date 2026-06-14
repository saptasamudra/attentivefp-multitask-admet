"""
GNN Baselines for MoleculeNet (GIN, GCN, GAT)
Following Professor's Phase 1 Priority Requirements

Matches D-MPNN experimental protocol:
- Same datasets (9 MoleculeNet)
- Same scaffold splits
- Same random seeds (0-4)
- Same evaluation metrics

Author: Sapta
Date: 2026-04-01
Lab: SAMLab, Guizhou University

FIXES (2026-04-01):
- Replaced BatchNorm1d with LayerNorm to fix single-atom batch crash
- Added drop_last=True to train_loader as additional safety
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_mean_pool, global_add_pool
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score, mean_squared_error
from collections import defaultdict
import argparse
import random
import json
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    'ESOL': {'task': 'regression', 'metric': 'RMSE', 'num_tasks': 1},
    'FreeSolv': {'task': 'regression', 'metric': 'RMSE', 'num_tasks': 1},
    'Lipo': {'task': 'regression', 'metric': 'RMSE', 'num_tasks': 1},
    'BACE': {'task': 'classification', 'metric': 'AUC', 'num_tasks': 1},
    'BBBP': {'task': 'classification', 'metric': 'AUC', 'num_tasks': 1},
    'HIV': {'task': 'classification', 'metric': 'AUC', 'num_tasks': 1},
    'ClinTox': {'task': 'classification', 'metric': 'AUC', 'num_tasks': 2},
    'Tox21': {'task': 'classification', 'metric': 'AUC', 'num_tasks': 12},
    'SIDER': {'task': 'classification', 'metric': 'AUC', 'num_tasks': 27},
}


# ============================================================================
# MOLECULAR FEATURIZATION (Matching D-MPNN style)
# ============================================================================

def atom_features(atom):
    """Generate atom feature vector"""
    return np.array([
        # Atom type (one-hot, 100 types max)
        *one_hot_encoding(atom.GetAtomicNum(), list(range(1, 101))),
        # Degree
        *one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]),
        # Formal charge
        *one_hot_encoding(atom.GetFormalCharge(), [-1, 0, 1]),
        # Hybridization
        *one_hot_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]),
        # Aromaticity
        1 if atom.GetIsAromatic() else 0,
        # Total num Hs
        atom.GetTotalNumHs(),
    ], dtype=np.float32)


def one_hot_encoding(value, choices):
    """One-hot encoding"""
    encoding = [0] * (len(choices) + 1)  # +1 for unknown
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def mol_to_graph(smiles):
    """Convert SMILES to PyG graph"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))
    x = torch.tensor(np.array(atom_feats), dtype=torch.float)

    # Edge indices
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])  # Undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# ============================================================================
# SCAFFOLD SPLIT (Matching D-MPNN protocol)
# ============================================================================

def generate_scaffold(smiles, include_chirality=False):
    """Generate Murcko scaffold for molecule"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_split(dataset, smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """Scaffold-based dataset splitting (balanced)"""
    np.random.seed(seed)

    # Group by scaffold
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        if scaffold is not None:
            scaffolds[scaffold].append(idx)

    # Sort scaffolds by size (descending)
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        (scaffold, set(indices)) for scaffold, indices in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    # Balanced split
    train_size = int(frac_train * len(dataset))
    valid_size = int(frac_valid * len(dataset))

    train_indices, valid_indices, test_indices = [], [], []

    for scaffold, indices in scaffold_sets:
        if len(train_indices) + len(indices) <= train_size:
            train_indices.extend(indices)
        elif len(valid_indices) + len(indices) <= valid_size:
            valid_indices.extend(indices)
        else:
            test_indices.extend(indices)

    print(f"Scaffold split (seed {seed}): train={len(train_indices)}, valid={len(valid_indices)}, test={len(test_indices)}")

    return train_indices, valid_indices, test_indices


# ============================================================================
# GNN MODEL ARCHITECTURES
# ============================================================================

class GINModel(nn.Module):
    """Graph Isomorphism Network"""
    def __init__(self, num_features, num_tasks, hidden_dim=300, num_layers=3, dropout=0.0, task_type='classification'):
        super(GINModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.task_type = task_type

        # GIN layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # LayerNorm instead of BatchNorm1d

        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            self.convs.append(GINConv(mlp))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Readout layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_add_pool(x, batch)

        # Readout
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class GCNModel(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, num_features, num_tasks, hidden_dim=300, num_layers=3, dropout=0.0, task_type='classification'):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.task_type = task_type

        # GCN layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # LayerNorm instead of BatchNorm1d

        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(num_features, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Readout layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Readout
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class GATModel(nn.Module):
    """Graph Attention Network"""
    def __init__(self, num_features, num_tasks, hidden_dim=300, num_layers=3, dropout=0.0, num_heads=4, task_type='classification'):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.task_type = task_type

        # GAT layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # LayerNorm instead of BatchNorm1d

        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(num_features, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
            elif i == num_layers - 1:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Readout layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Readout
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, loader, optimizer, device, task_type):
    """Single training epoch"""
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)

        # Handle missing labels
        is_valid = ~torch.isnan(data.y)

        if task_type == 'classification':
            loss = F.binary_cross_entropy_with_logits(out[is_valid], data.y[is_valid])
        else:  # regression
            loss = F.mse_loss(out[is_valid], data.y[is_valid])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, task_type, num_tasks):
    """Evaluate model"""
    model.eval()
    y_true_all = [[] for _ in range(num_tasks)]
    y_pred_all = [[] for _ in range(num_tasks)]

    for data in loader:
        data = data.to(device)
        out = model(data)

        if task_type == 'classification':
            out = torch.sigmoid(out)

        # Collect predictions per task
        for task_idx in range(num_tasks):
            is_valid = ~torch.isnan(data.y[:, task_idx])
            if is_valid.sum() > 0:
                y_true_all[task_idx].extend(data.y[is_valid, task_idx].cpu().numpy())
                y_pred_all[task_idx].extend(out[is_valid, task_idx].cpu().numpy())

    # Calculate metrics per task
    metrics = []
    for task_idx in range(num_tasks):
        if len(y_true_all[task_idx]) == 0:
            metrics.append(np.nan)
            continue

        y_true = np.array(y_true_all[task_idx])
        y_pred = np.array(y_pred_all[task_idx])

        if task_type == 'classification':
            if len(np.unique(y_true)) == 1:
                metrics.append(np.nan)
            else:
                metrics.append(roc_auc_score(y_true, y_pred))
        else:  # regression
            metrics.append(np.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = np.array(metrics)
    return np.nanmean(metrics) if not np.all(np.isnan(metrics)) else np.nan


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def run_experiment(args):
    """Run single experiment (one model, one dataset, one seed)"""

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Load dataset configuration
    config = DATASET_CONFIG[args.dataset]
    task_type = config['task']
    num_tasks = config['num_tasks']

    # Load dataset
    data_path = os.path.join(args.data_dir, f"{args.dataset}.csv")
    df = pd.read_csv(data_path)

    # Extract SMILES and labels
    smiles_list = df['smiles'].tolist()

    if num_tasks == 1:
        labels = df.iloc[:, 1].values.reshape(-1, 1).astype(np.float32)
    else:
        labels = df.iloc[:, 1:1+num_tasks].values.astype(np.float32)

    # Convert to graphs
    print("Converting molecules to graphs...")
    graphs = []
    valid_indices = []
    for idx, smiles in enumerate(tqdm(smiles_list)):
        graph = mol_to_graph(smiles)
        if graph is not None:
            graph.y = torch.tensor(labels[idx], dtype=torch.float).view(1, -1)
            graphs.append(graph)
            valid_indices.append(idx)

    print(f"Valid molecules: {len(graphs)} / {len(smiles_list)}")

    # Scaffold split
    valid_smiles = [smiles_list[i] for i in valid_indices]
    train_idx, valid_idx, test_idx = scaffold_split(
        graphs, valid_smiles,
        frac_train=0.8, frac_valid=0.1, frac_test=0.1,
        seed=args.seed
    )

    train_data = [graphs[i] for i in train_idx]
    valid_data = [graphs[i] for i in valid_idx]
    test_data = [graphs[i] for i in test_idx]

    # Create dataloaders — drop_last=True prevents single-sample batches crashing LayerNorm
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    num_features = graphs[0].x.shape[1]

    if args.model == 'GIN':
        model = GINModel(num_features, num_tasks, hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers, dropout=args.dropout, task_type=task_type)
    elif args.model == 'GCN':
        model = GCNModel(num_features, num_tasks, hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers, dropout=args.dropout, task_type=task_type)
    elif args.model == 'GAT':
        model = GATModel(num_features, num_tasks, hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers, dropout=args.dropout,
                        num_heads=args.num_heads, task_type=task_type)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_valid_metric = float('inf') if task_type == 'regression' else 0.0
    best_epoch = 0
    patience_counter = 0

    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, task_type)
        valid_metric = evaluate(model, valid_loader, device, task_type, num_tasks)

        # Check improvement
        improved = False
        if task_type == 'regression':
            if valid_metric < best_valid_metric:
                best_valid_metric = valid_metric
                best_epoch = epoch
                improved = True
        else:
            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                best_epoch = epoch
                improved = True

        if improved:
            patience_counter = 0
            save_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_seed{args.seed}_best.pt")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            metric_name = config['metric']
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Valid {metric_name}: {valid_metric:.4f} | Best: {best_valid_metric:.4f} (epoch {best_epoch})")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    best_model_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_seed{args.seed}_best.pt")
    model.load_state_dict(torch.load(best_model_path))

    test_metric = evaluate(model, test_loader, device, task_type, num_tasks)

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"Best Valid {config['metric']}: {best_valid_metric:.4f} (epoch {best_epoch})")
    print(f"Test {config['metric']}: {test_metric:.4f}")
    print(f"{'='*80}\n")

    results = {
        'model': args.model,
        'dataset': args.dataset,
        'seed': args.seed,
        'task_type': task_type,
        'metric': config['metric'],
        'best_valid': float(best_valid_metric),
        'test': float(test_metric),
        'best_epoch': best_epoch
    }

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='GNN Baselines (GIN/GCN/GAT) for MoleculeNet')

    parser.add_argument('--model', type=str, required=True, choices=['GIN', 'GCN', 'GAT'])
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./baselines/saved_models')
    parser.add_argument('--results_file', type=str, default='./baselines/results/gnn_results.json')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    results = run_experiment(args)

    if os.path.exists(args.results_file):
        with open(args.results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(results)

    with open(args.results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {args.results_file}")


if __name__ == '__main__':
    main()
