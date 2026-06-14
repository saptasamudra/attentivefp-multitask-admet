"""
Phase 1.2: Generic MoE Wrapper Module
=====================================
Wrap any GNN encoder with MoE to prove "MoE is plug-and-play enhancement"

Usage:
  encoder = GIN(...)  # or GCN, GAT, etc.
  moe_router = MixtureOfExperts(input_dim=200, expert_hidden=200, num_experts=4, top_k=2)
  x_moe, aux_loss, routing = moe_router(encoder_output)

Then concatenate encoder output + moe output and feed to task heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureOfExperts(nn.Module):
    """
    Sparse top-k MoE routing for any GNN embedding.
    
    Input: [batch_size, hidden_dim] molecular embeddings from any encoder
    Output: [batch_size, hidden_dim] expert-routed representations
    
    Args:
        input_dim (int): Dimension of input embeddings
        expert_hidden (int): Hidden dimension of expert networks
        num_experts (int): Number of experts
        top_k (int): Number of top experts to route each molecule to
        dropout (float): Dropout rate
        load_balance_weight (float): Weight for load-balancing auxiliary loss
    """
    
    def __init__(self, input_dim, expert_hidden, num_experts, top_k, dropout=0.2, load_balance_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        # Gate network: maps embedding to expert routing scores
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        
        # Experts: identical structure, each is a 2-layer MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, input_dim),
            )
            for _ in range(num_experts)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        
        Returns:
            expert_out: [batch_size, input_dim] — routed embeddings
            aux_loss: scalar — load-balancing loss (add to training loss)
            gate_soft: [batch_size, num_experts] — routing probabilities
        """
        B = x.size(0)
        
        # ─ Gate routing ─
        gate_logits = self.gate(x)  # [B, num_experts]
        gate_soft = F.softmax(gate_logits, dim=-1)  # [B, num_experts]
        
        # Load-balancing auxiliary loss: penalize unbalanced expert usage
        expert_usage = gate_soft.mean(dim=0)  # [num_experts] — avg activation per expert
        ideal = torch.ones_like(expert_usage) / self.num_experts
        aux_loss = ((expert_usage - ideal) ** 2).mean()
        
        # ─ Sparse top-k selection ─
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)  # [B, top_k]
        routing_weights = F.softmax(topk_vals, dim=-1)  # normalize top-k scores
        
        # ─ Expert computation (sparse) ─
        expert_out = torch.zeros(B, x.size(-1), device=x.device, dtype=x.dtype)
        
        for ei in range(self.num_experts):
            # Find which molecules route to expert ei
            mask = (topk_idx == ei).any(dim=-1)  # [B]
            if not mask.any():
                continue
            
            # Compute expert output for selected molecules
            out = self.experts[ei](self.dropout(x[mask]))  # [num_selected, input_dim]
            
            # Weight by routing probability for this expert
            pos_mask = (topk_idx == ei)  # [B, top_k] — where expert ei appears in top-k
            w = (routing_weights * pos_mask.float()).sum(dim=-1)  # [B] — weight for each molecule
            
            # Accumulate weighted expert output
            expert_out[mask] += out * w[mask].unsqueeze(-1)
        
        return expert_out, aux_loss, gate_soft


class GenericMoEModel(nn.Module):
    """
    Wrapper: Any GNN encoder + MoE + task heads
    
    Flow:
      x, edge_index, edge_attr, batch
        -> encoder           -> mol_repr [B, hidden_dim]
        -> MoE               -> expert_repr [B, hidden_dim]
        -> cat + fusion      -> fused [B, 2*hidden_dim]
        -> task_heads        -> predictions
    """
    
    def __init__(self, encoder, input_dim, hidden_dim, num_experts, top_k, datasets_dict, use_moe=True):
        """
        Args:
            encoder: Pre-initialized GNN model (GIN, GCN, GAT, etc.)
            input_dim: Atom feature dimension
            hidden_dim: Hidden dimension
            num_experts: Number of MoE experts
            top_k: Top-k routing
            datasets_dict: DATASETS config dict with 'num_tasks' per dataset
            use_moe: If False, skip MoE and use plain encoder (for ablation)
        """
        super().__init__()
        self.encoder = encoder
        self.use_moe = use_moe
        self.hidden_dim = hidden_dim
        
        if use_moe:
            self.moe = MixtureOfExperts(
                input_dim=hidden_dim,
                expert_hidden=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                dropout=0.2,
                load_balance_weight=0.01,
            )
            fused_dim = hidden_dim * 2
        else:
            self.moe = None
            fused_dim = hidden_dim
        
        # Task-specific heads
        self.heads = nn.ModuleDict({
            name: nn.Linear(fused_dim, cfg['num_tasks'])
            for name, cfg in datasets_dict.items()
        })
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Returns:
            preds: dict {dataset_name: predictions}
            aux_loss: scalar (0 if use_moe=False)
        """
        # Encoder
        mol_repr = self.dropout(self.encoder(x, edge_index))  # [B, hidden_dim]
        
        if self.use_moe:
            expert_repr, aux_loss, _ = self.moe(mol_repr)
            fused = self.dropout(torch.cat([mol_repr, expert_repr], dim=-1))
        else:
            fused = mol_repr
            aux_loss = torch.tensor(0.0, device=x.device)
        
        # Task heads
        preds = {name: head(fused) for name, head in self.heads.items()}
        
        return preds, aux_loss


# ─ Example usage ─
if __name__ == '__main__':
    from torch_geometric.nn.models import GIN, GCN, GAT
    
    # Create a base GIN encoder
    gin = GIN(in_channels=39, hidden_channels=200, num_layers=2, out_channels=200, dropout=0.2)
    
    # Wrap with MoE
    datasets = {
        'ESOL': {'num_tasks': 1},
        'BACE': {'num_tasks': 1},
    }
    
    model_with_moe = GenericMoEModel(
        encoder=gin,
        input_dim=39,
        hidden_dim=200,
        num_experts=4,
        top_k=2,
        datasets_dict=datasets,
        use_moe=True
    )
    
    # Dummy input
    batch_x = torch.randn(32, 39)
    batch_edge_index = torch.randint(0, 32, (2, 64))
    batch = torch.zeros(32, dtype=torch.long)
    
    preds, aux_loss = model_with_moe(batch_x, batch_edge_index, None, batch)
    print("Predictions keys:", list(preds.keys()))
    print("Aux loss:", aux_loss.item())
