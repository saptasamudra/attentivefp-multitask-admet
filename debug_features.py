# Run this to find the exact feature count your GenFeatures produces
# and compare with what the model expects

from rdkit import Chem
import torch

# Copy of GenFeatures from attentivefp_moe.py
class GenFeatures:
    def __init__(self):
        self.symbols = [
            'B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other'
        ]
        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other',
        ]
        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]

    def __call__(self, data):
        mol = Chem.MolFromSmiles(data.smiles)
        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())
                   if atom.GetSymbol() in self.symbols else -1] = 1.
            degree = [0.] * 6
            degree[min(atom.GetDegree(), 5)] = 1.
            formal_charge  = atom.GetFormalCharge()
            radical_e      = atom.GetNumRadicalElectrons()
            hybridization  = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(
                atom.GetHybridization())
                if atom.GetHybridization() in self.hybridizations else -1] = 1.
            aromaticity    = 1. if atom.GetIsAromatic() else 0.
            hydrogens      = [0.] * 5
            hydrogens[min(atom.GetTotalNumHs(), 4)] = 1.
            xs.append(symbol + degree + [formal_charge, radical_e] +
                      hybridization + [aromaticity] + hydrogens)
        data.x = torch.tensor(xs, dtype=torch.float)
        return data

# Test on a simple molecule
class FakeData:
    smiles = 'CC(=O)O'  # acetic acid

data = FakeData()
gf = GenFeatures()
gf(data)

vec = data.x[0].tolist()
print(f"Total features per atom: {len(vec)}")
print(f"  symbols       : {len(gf.symbols)} features")
print(f"  degree        : 6 features")
print(f"  formal_charge : 1 feature")
print(f"  radical_e     : 1 feature")
print(f"  hybridization : {len(gf.hybridizations)} features")
print(f"  aromaticity   : 1 feature")
print(f"  hydrogens     : 5 features")
total = len(gf.symbols) + 6 + 1 + 1 + len(gf.hybridizations) + 1 + 5
print(f"  Expected total: {total}")

# Now check what your ESOL cached data has
from torch_geometric.datasets import MoleculeNet
ds = MoleculeNet(root='data/ESOL', name='ESOL')
print(f"\nCached ESOL features: {ds[0].x.shape[1]}")
print(f"Match: {ds[0].x.shape[1] == len(vec)}")
