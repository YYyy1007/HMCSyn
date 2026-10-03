
import os
import warnings
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from torch_geometric.data import Data

# Disable RDKit logs and deprecation warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=DeprecationWarning)

BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


# ------------ Utility functions ------------
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    x = x if x in allowable_set else allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
             'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
             'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
             'Pt', 'Hg', 'Pb', 'Unknown']
        ) +
        one_of_k_encoding(atom.GetDegree(), list(range(11))) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11))) +
        one_of_k_encoding_unk(atom.GetTotalValence(), list(range(11))) +
        [atom.GetIsAromatic()]
    )


def bond_features(bond):
    return np.array(
        one_of_k_encoding(bond.GetBondType(), BOND_LIST) +
        one_of_k_encoding(bond.GetBondDir(), BONDDIR_LIST)
    )


def smiles_to_pyg_graph(smiles):
    """Convert a SMILES string to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / (sum(feature) + 1e-6))

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        feature = bond_features(bond)
        edge_feat.append(feature / (sum(feature) + 1e-6))
        edge_feat.append(feature / (sum(feature) + 1e-6))

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float)
    x = torch.tensor(np.array(features), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def detect_and_read_csv(file_path):
    """Automatically detect encoding and read a CSV file."""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'utf-8-sig', 'cp936']

    for encoding in encodings:
        try:
            print(f"Trying to read file with encoding {encoding}: {file_path}")
            data = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read file with encoding {encoding}")
            return data
        except UnicodeDecodeError as e:
            print(f"Encoding {encoding} failed: {e}")
            continue
        except Exception as e:
            print(f"Error while reading with encoding {encoding}: {e}")
            continue

    raise ValueError(f"Cannot read file with any encoding: {file_path}")


# ------------ Main process ------------
if __name__ == "__main__":
    root = 'data'
    out_dir = os.path.join(root, 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Read drug information (columns: names, smiles)
    try:
        data = detect_and_read_csv(os.path.join(root, 'drugInfo.csv'))
    except Exception as e:
        print(f"Failed to read drugInfo.csv: {e}")
        print("Please check if the file exists or manually specify the correct encoding.")
        exit(1)

    graphs = {}
    gen = GetMorganGenerator(radius=2, fpSize=1024)

    success_count = 0
    error_count = 0

    for i in range(data.shape[0]):
        try:
            name = data['names'][i]
            smiles = data['smiles'][i]

            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Invalid SMILES string: {smiles} (drug: {name})")
                error_count += 1
                continue

            g = smiles_to_pyg_graph(smiles)
            fp = torch.tensor(gen.GetFingerprint(mol), dtype=torch.float)

            graphs[name] = {
                'smiles': smiles,
                'graph': g,
                'fp': fp
            }
            success_count += 1

            if success_count % 100 == 0:
                print(f"Processed {success_count} drugs...")

        except Exception as e:
            print(f"Error processing drug {name}: {e}")
            error_count += 1
            continue

    # Save to data/output/drug_pyg.pkl
    try:
        with open(os.path.join(out_dir, 'drug_pyg.pkl'), 'wb') as f:
            pd.to_pickle(graphs, f)

        print(f'Processing completed! Success: {success_count}, Failed: {error_count}')
        print('Output file: ' + os.path.join(out_dir, 'drug_pyg.pkl'))

    except Exception as e:
        print(f"Failed to save file: {e}")
        # Fallback: save as PyTorch format
        try:
            torch.save(graphs, os.path.join(out_dir, 'drug_pyg.pt'))
            print('Saved as drug_pyg.pt (PyTorch format)')
        except Exception as e2:
            print(f"All saving methods failed: {e2}")