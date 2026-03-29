from torch_geometric.datasets import MoleculeNet

names = ['ESOL','FreeSolv','Lipo','BACE','BBBP','HIV','ClinTox','Tox21','SIDER']
bad = []
for name in names:
    ds = MoleculeNet(root='data/'+name, name=name)
    shape = ds[0].x.shape[1]
    status = "OK" if shape == 39 else f"BAD ({shape})"
    print(f"{name:<12} features={shape}  {status}")
    if shape != 39:
        bad.append(name)

print()
if bad:
    print("Delete these folders and rerun attentivefp_moe.py:")
    for name in bad:
        print(f"  rmdir /s /q D:\\molprop_project\\data\\{name}")
else:
    print("All 39 features. Safe to run.")
