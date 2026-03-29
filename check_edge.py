from torch_geometric.datasets import MoleculeNet

for name in ['ESOL', 'BACE', 'Tox21']:
    ds = MoleculeNet(root='data/'+name, name=name)
    d = ds[0]
    print(f"{name}: x.shape={d.x.shape}, edge_attr.shape={d.edge_attr.shape}")
