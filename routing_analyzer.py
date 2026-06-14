"""
Phase 1.3: Expert Routing Interpretability Analysis
===================================================
Analyze MoE routing patterns to understand expert specialization.

Outputs:
1. Expert utilization statistics (% of molecules routed to each expert)
2. t-SNE/UMAP visualization of routing by molecular properties
3. Dataset-specific expert distribution heatmap
4. Chemical property correlation with expert preference
"""

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen


class RoutingAnalyzer:
    """Analyze MoE routing decisions for interpretability."""
    
    def __init__(self, model, test_loaders, num_experts=4, device='cuda'):
        """
        Args:
            model: MoE model with routing (must return gate_soft)
            test_loaders: dict {dataset_name: DataLoader}
            num_experts: K value
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.test_loaders = test_loaders
        self.num_experts = num_experts
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def collect_routing_decisions(self):
        """
        Collect routing decisions and molecular features across all datasets.
        
        Returns:
            routing_data: dict {
                'gate_probs': [N, num_experts],
                'top_expert': [N],
                'molecule_ids': [N],
                'dataset_names': [N],
                'smiles': [N],
            }
        """
        all_gate_probs = []
        all_top_experts = []
        all_smiles = []
        all_datasets = []
        
        for ds_name, loader in self.test_loaders.items():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Forward pass — need to extract gate outputs
                # Modified model should return gate_soft alongside predictions
                preds, aux, gate_soft = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                all_gate_probs.append(gate_soft.cpu().numpy())
                all_top_experts.append(gate_soft.argmax(dim=-1).cpu().numpy())
                
                if hasattr(batch, 'smiles'):
                    all_smiles.extend(batch.smiles)
                
                all_datasets.extend([ds_name] * batch.x.size(0))
        
        routing_data = {
            'gate_probs': np.concatenate(all_gate_probs, axis=0),
            'top_expert': np.concatenate(all_top_experts, axis=0),
            'dataset_names': np.array(all_datasets),
            'smiles': all_smiles if all_smiles else None,
        }
        
        return routing_data
    
    def expert_utilization_stats(self, routing_data):
        """
        Compute expert usage statistics.
        
        Returns:
            stats: dict {
                'utilization': [num_experts] — % of molecules routed to each expert,
                'per_dataset_util': {dataset_name: [num_experts]},
                'entropy': float — routing entropy (1.0 = uniform, 0 = one expert),
            }
        """
        gate_probs = routing_data['gate_probs']
        top_experts = routing_data['top_expert']
        dataset_names = routing_data['dataset_names']
        
        # Global utilization
        utilization = np.bincount(top_experts, minlength=self.num_experts) / len(top_experts)
        
        # Per-dataset utilization
        per_dataset = {}
        for ds in np.unique(dataset_names):
            mask = dataset_names == ds
            per_dataset[ds] = np.bincount(
                top_experts[mask], minlength=self.num_experts) / mask.sum()
        
        # Entropy (higher = more balanced)
        entropy = -np.sum(gate_probs.mean(axis=0) * np.log(gate_probs.mean(axis=0) + 1e-8))
        
        return {
            'utilization': utilization,
            'per_dataset_util': per_dataset,
            'entropy': entropy,
        }
    
    def compute_molecular_features(self, routing_data):
        """
        Extract chemical properties from SMILES.
        Correlate with expert preference.
        """
        if routing_data['smiles'] is None:
            print("SMILES not available in batch data.")
            return None
        
        smiles_list = routing_data['smiles']
        top_experts = routing_data['top_expert']
        
        features = {
            'mw': [],
            'logp': [],
            'hbd': [],
            'hba': [],
            'rotatable_bonds': [],
            'expert': [],
        }
        
        for smi, exp in zip(smiles_list, top_experts):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            
            features['mw'].append(Descriptors.MolWt(mol))
            features['logp'].append(Crippen.MolLogP(mol))
            features['hbd'].append(Descriptors.NumHDonors(mol))
            features['hba'].append(Descriptors.NumHAcceptors(mol))
            features['rotatable_bonds'].append(Descriptors.NumRotatableBonds(mol))
            features['expert'].append(exp)
        
        return features
    
    def visualize_routing_tsne(self, routing_data, output_path='routing_tsne.png'):
        """t-SNE visualization of expert routing."""
        gate_probs = routing_data['gate_probs']
        top_experts = routing_data['top_expert']
        datasets = routing_data['dataset_names']
        
        # PCA first for speed
        pca = PCA(n_components=50)
        gate_reduced = pca.fit_transform(gate_probs)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        gate_2d = tsne.fit_transform(gate_reduced)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Color by top expert
        scatter1 = axes[0].scatter(gate_2d[:, 0], gate_2d[:, 1], c=top_experts, cmap='tab10', alpha=0.7, s=20)
        axes[0].set_title('Routing by Top Expert')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter1, ax=axes[0], label='Expert ID')
        
        # Color by dataset
        dataset_map = {ds: i for i, ds in enumerate(np.unique(datasets))}
        dataset_colors = np.array([dataset_map[ds] for ds in datasets])
        scatter2 = axes[1].scatter(gate_2d[:, 0], gate_2d[:, 1], c=dataset_colors, cmap='tab20', alpha=0.7, s=20)
        axes[1].set_title('Routing by Dataset')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=axes[1], label='Dataset')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()
    
    def visualize_expert_utilization_heatmap(self, routing_data, output_path='expert_util_heatmap.png'):
        """Heatmap of expert utilization per dataset."""
        stats = self.expert_utilization_stats(routing_data)
        per_dataset_util = stats['per_dataset_util']
        
        datasets = sorted(per_dataset_util.keys())
        util_matrix = np.array([per_dataset_util[ds] for ds in datasets])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(util_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=[f'Expert {i}' for i in range(self.num_experts)],
                    yticklabels=datasets, ax=ax, cbar_kws={'label': 'Utilization %'})
        ax.set_title('Expert Utilization Heatmap (per Dataset)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()
    
    def print_stats(self, routing_data):
        """Print interpretability statistics to console."""
        stats = self.expert_utilization_stats(routing_data)
        
        print("\n" + "="*70)
        print("  EXPERT ROUTING ANALYSIS")
        print("="*70)
        print(f"\nGlobal Expert Utilization:")
        for i, util in enumerate(stats['utilization']):
            print(f"  Expert {i}: {util*100:.2f}%")
        print(f"\nRouting Entropy (max={np.log(self.num_experts):.3f}): {stats['entropy']:.3f}")
        
        print(f"\nPer-Dataset Expert Utilization:")
        for ds, util in stats['per_dataset_util'].items():
            print(f"  {ds}:")
            for i, u in enumerate(util):
                print(f"    Expert {i}: {u*100:.2f}%")
        
        # Molecular features correlation
        features = self.compute_molecular_features(routing_data)
        if features:
            print(f"\nMolecular Properties vs Expert Preference:")
            for prop in ['mw', 'logp', 'hbd', 'hba', 'rotatable_bonds']:
                per_expert_prop = {i: [] for i in range(self.num_experts)}
                for val, exp in zip(features[prop], features['expert']):
                    per_expert_prop[exp].append(val)
                
                means = [np.mean(per_expert_prop[i]) if per_expert_prop[i] else 0
                        for i in range(self.num_experts)]
                print(f"  {prop}: {means}")


# ─ Example usage ─
if __name__ == '__main__':
    print("RoutingAnalyzer module loaded.")
    print("Usage in your experiment:")
    print("""
    analyzer = RoutingAnalyzer(model, test_loaders, num_experts=4, device=device)
    routing_data = analyzer.collect_routing_decisions()
    analyzer.print_stats(routing_data)
    analyzer.visualize_routing_tsne(routing_data)
    analyzer.visualize_expert_utilization_heatmap(routing_data)
    """)
