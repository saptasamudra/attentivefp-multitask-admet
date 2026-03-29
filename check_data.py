from torch_geometric.datasets import MoleculeNet

print('Checking HIV...')
ds = MoleculeNet(root='data/HIV', name='HIV')
print(f'HIV OK: {len(ds)} molecules, {ds[0].y.shape} labels')

print('Checking SIDER...')
ds2 = MoleculeNet(root='data/SIDER', name='SIDER')
print(f'SIDER OK: {len(ds2)} molecules, {ds2[0].y.shape[1]} tasks')

print('Both datasets ready. Safe to run attentivefp_moe.py')
