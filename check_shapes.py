from torch_geometric.datasets import MoleculeNet

names = ['ESOL','FreeSolv','Lipo','BACE','BBBP','HIV','ClinTox','Tox21','SIDER']
for name in names:
    ds = MoleculeNet(root='data/'+name, name=name)
    print(f"{name:<12} x.shape = {ds[0].x.shape}")
