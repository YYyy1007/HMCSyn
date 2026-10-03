import os, glob, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
import joblib

# ==================== Molecular graph construction ====================
BOND_LIST = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
             rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]
BONDDIR_LIST = [rdchem.BondDir.NONE, rdchem.BondDir.ENDUPRIGHT, rdchem.BondDir.ENDDOWNRIGHT]


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


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")
    features = []
    for atom in mol.GetAtoms():
        feat = atom_features(atom)
        features.append(feat / (sum(feat) + 1e-6))
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        feat = bond_features(bond)
        edge_feat.append(feat / (sum(feat) + 1e-6))
        edge_feat.append(feat / (sum(feat) + 1e-6))
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float)
    x = torch.tensor(np.array(features), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ==================== SDAE model ====================
class Encoder(nn.Module):
    def __init__(self, device, indim, outdim=512):
        super().__init__()
        self.linear1 = nn.Linear(indim, 2048, device=device)
        self.linear2 = nn.Linear(2048, 1024, device=device)
        self.linear3 = nn.Linear(1024, outdim, device=device)

    def forward(self, x):
        x = torch.selu(self.linear1(x))
        x = torch.selu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, device, outdim, indim=512):
        super().__init__()
        self.linear3 = nn.Linear(indim, 1024, device=device)
        self.linear2 = nn.Linear(1024, 2048, device=device)
        self.linear1 = nn.Linear(2048, outdim, device=device)

    def forward(self, x):
        x = torch.selu(self.linear3(x))
        x = torch.selu(self.linear2(x))
        x = torch.sigmoid(self.linear1(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, device, indim, outdim=512):
        super().__init__()
        self.encoder = Encoder(device, indim, outdim)
        self.decoder = Decoder(device, outdim=indim, indim=outdim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_sdae(features, device, lr=1e-4, epochs=2500, batch_size=30, patience=100):
    model = AutoEncoder(device, features.shape[1], 512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    feat_list = features.tolist()
    random.seed(42)
    train_list = random.sample(feat_list, int(0.9 * len(feat_list)))
    test_list = [item for item in feat_list if item not in train_list]
    train_t = torch.tensor(train_list, dtype=torch.float32, device=device)
    test_t = torch.tensor(test_list, dtype=torch.float32, device=device)
    train_loader = DataLoader(TensorDataset(train_t), batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    best_state = None
    no_improve = 0
    for epoch in range(1, epochs + 1):
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            encoded, decoded = model(x_batch)
            loss = loss_fn(decoded, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            encoded, decoded = model(test_t)
            test_loss = loss_fn(decoded, test_t).item()

        if test_loss < best_loss:
            best_loss = test_loss
            best_state = model.encoder.state_dict()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is None:
        print("Warning: SDAE training did not improve, using last epoch's state.")
        best_state = model.encoder.state_dict()

    final_encoder = Encoder(device, features.shape[1], 512).to(device)
    final_encoder.load_state_dict(best_state)
    return final_encoder


def fill_nan_with_zero(X):
    X = np.where(np.isfinite(X), X, 0.0)
    return X


# ==================== Load features by ACH-ID ====================
def load_selected_cells_by_ach(data_root, ach_ids):
    """
    Extract expression, CNV, and pathway features from raw files for the given ACH-ID list.
    Pathway column names support substring matching to handle naming differences
    (e.g., MSTO-211H vs MSTO).
    """
    import re
    print("\n=== DEBUG: Starting load_selected_cells_by_ach ===")
    print(f"Total ACH IDs provided: {len(ach_ids)}")
    print(f"First 5 ACHs: {ach_ids[:5]}")

    # Load Model.csv
    model_file = os.path.join(data_root, 'Model.csv')
    if not os.path.exists(model_file):
        raise FileNotFoundError("Model.csv is required!")
    model_df = pd.read_csv(model_file)
    print(f"Model.csv columns: {model_df.columns.tolist()}")

    ach_to_fullname = dict(zip(model_df['ModelID'], model_df['CellLineName']))

    print("\n=== Debug: ACH to FullName mapping (first 5) ===")
    for ach in ach_ids[:5]:
        full = ach_to_fullname.get(ach)
        print(f"  {ach} -> Full: {full}")

    # ---------- Load expression data ----------
    expr_file = os.path.join(data_root, 'OmicsExpressionTPMLogp1HumanProteinCoding.csv')
    if not os.path.exists(expr_file):
        candidates = glob.glob(os.path.join(data_root, 'OmicsExpressionTPMLogp1Human*.csv'))
        if candidates:
            expr_file = candidates[0]
        else:
            raise FileNotFoundError("No OmicsExpression file found")
    expr_df = pd.read_csv(expr_file, dtype=str, low_memory=False)
    if 'ModelID' not in expr_df.columns:
        raise RuntimeError("No 'ModelID' column found in expression file.")
    expr_df.set_index('ModelID', inplace=True)
    numeric_cols = []
    for col in expr_df.columns:
        try:
            pd.to_numeric(expr_df[col], errors='raise')
            numeric_cols.append(col)
        except:
            pass
    if len(numeric_cols) == 0:
        raise RuntimeError("No numeric columns found in expression file.")
    expr_df = expr_df[numeric_cols].astype(float)
    print(f"\nExpression data: {expr_df.shape}, {len(numeric_cols)} numeric columns")
    print(f"Expression index first 5: {expr_df.index[:5].tolist()}")

    # ---------- Load CNV data ----------
    cnv_file = os.path.join(data_root, 'OmicsCNGeneWGS.csv')
    if not os.path.exists(cnv_file):
        raise FileNotFoundError(f"Missing {cnv_file}")
    cnv_df = pd.read_csv(cnv_file, dtype=str, low_memory=False)
    if 'ModelID' not in cnv_df.columns:
        raise RuntimeError("No 'ModelID' column found in CNV file.")
    cnv_df.set_index('ModelID', inplace=True)
    numeric_cols_cnv = []
    for col in cnv_df.columns:
        try:
            pd.to_numeric(cnv_df[col], errors='raise')
            numeric_cols_cnv.append(col)
        except:
            pass
    if len(numeric_cols_cnv) == 0:
        raise RuntimeError("No numeric columns found in CNV file.")
    cnv_df = cnv_df[numeric_cols_cnv].astype(float)
    print(f"\nCNV data: {cnv_df.shape}, {len(numeric_cols_cnv)} numeric columns")
    print(f"CNV index first 5: {cnv_df.index[:5].tolist()}")

    # ---------- Load pathway data ----------
    pw_file = os.path.join(data_root, 'ONeil_31_gsva_1329_dim.csv')
    if not os.path.exists(pw_file):
        raise FileNotFoundError(f"Missing {pw_file}")
    pw_df = pd.read_csv(pw_file, dtype=str, low_memory=False)
    first_col = pw_df.columns[0]
    print(f"\nPathway file first column: {first_col}")
    print(f"Pathway file columns (first 10): {pw_df.columns[:10].tolist()}")
    pw_df.set_index(first_col, inplace=True)
    pw_df.columns = [c.strip().upper() for c in pw_df.columns]
    pw_df = pw_df.apply(pd.to_numeric, errors='coerce').astype(float)
    print(f"Pathway data shape after processing: {pw_df.shape}")
    print(f"Pathway columns (cells) first 10: {pw_df.columns[:10].tolist()}")

    # ---------- Filtering: exact match + substring match ----------
    valid_achs = []
    print("\n=== Debug: Filtering ACHs ===")
    for ach in ach_ids:
        if ach not in expr_df.index:
            print(f"  [skip] {ach} not in expression data")
            continue
        if ach not in cnv_df.index:
            print(f"  [skip] {ach} not in CNV data")
            continue
        fullname = ach_to_fullname.get(ach)
        if fullname is None:
            print(f"  [skip] {ach} has no full name in Model.csv")
            continue

        # Clean: remove all non-alphanumeric characters and convert to uppercase
        fullname_clean = re.sub(r'[^A-Z0-9]', '', fullname.upper())

        # Exact match
        if fullname_clean in pw_df.columns:
            matched_col = fullname_clean
            print(f"  [OK]   {ach} -> {matched_col}")
        else:
            # Try substring match
            matched_col = None
            for col in pw_df.columns:
                if fullname_clean in col or col in fullname_clean:
                    matched_col = col
                    break
            if matched_col is None:
                print(
                    f"  [skip] {ach} -> fullname '{fullname}' cleaned '{fullname_clean}' not found in pathway columns")
                continue
            else:
                print(f"  [OK]   {ach} -> {fullname_clean} (substring matched to {matched_col})")
        valid_achs.append(ach)

    if len(valid_achs) == 0:
        print("\nDEBUG: Pathway columns sample: ", pw_df.columns[:10].tolist())
        raise RuntimeError("No cells passed all three data source checks.")

    # ---------- Extract features ----------
    ge_feat = np.zeros((len(valid_achs), expr_df.shape[1]), dtype=np.float32)
    cnv_feat = np.zeros((len(valid_achs), cnv_df.shape[1]), dtype=np.float32)
    pw_feat = np.zeros((len(valid_achs), pw_df.shape[0]), dtype=np.float32)

    for i, ach in enumerate(valid_achs):
        ge_feat[i] = expr_df.loc[ach].values.astype(np.float32)
        cnv_feat[i] = cnv_df.loc[ach].values.astype(np.float32)
        fullname = ach_to_fullname[ach]
        fullname_clean = re.sub(r'[^A-Z0-9]', '', fullname.upper())

        # Use the same logic again to obtain the matched column name
        if fullname_clean in pw_df.columns:
            matched_col = fullname_clean
        else:
            for col in pw_df.columns:
                if fullname_clean in col or col in fullname_clean:
                    matched_col = col
                    break
        pw_feat[i] = pw_df[matched_col].values.astype(np.float32)

    print(f"\n=== Debug: Successfully loaded {len(valid_achs)} cells.")
    return ge_feat, cnv_feat, pw_feat, valid_achs


# ==================== Get or train SDAE (by ACH) ====================
def get_or_train_sdae_by_ach(data_root, ach_ids, device, cfg):
    """
    Generate or load dimension-reduced features based on the ACH-ID list.
    Returns: exp_feat, cnv_feat, path_feat, valid_ach_ids
    """
    exp_path = os.path.join(data_root, 'exp_512.npy')
    cnv_path = os.path.join(data_root, 'cnv_512.npy')
    path_path = os.path.join(data_root, 'path_1329.npy')
    ach_list_path = os.path.join(data_root, 'final_valid_achs.txt')

    need_regen = False
    if os.path.exists(exp_path) and os.path.exists(cnv_path) and os.path.exists(path_path):
        try:
            exp = np.load(exp_path)
            if exp.shape[0] != len(ach_ids):
                need_regen = True
        except:
            need_regen = True
    else:
        need_regen = True

    if need_regen:
        print(f"\nGenerating new features for {len(ach_ids)} cells using SDAE...")
        ge_feat, cnv_feat, pw_feat, valid_achs = load_selected_cells_by_ach(data_root, ach_ids)
        print(f"After filtering, {len(valid_achs)} cells remain.")

        # Fill missing values
        ge_feat = fill_nan_with_zero(ge_feat)
        cnv_feat = fill_nan_with_zero(cnv_feat)
        pw_feat = fill_nan_with_zero(pw_feat)

        # Normalize
        scaler_ge = MinMaxScaler()
        scaler_cnv = MinMaxScaler()
        scaler_path = StandardScaler()

        ge_norm = scaler_ge.fit_transform(ge_feat)
        cnv_norm = scaler_cnv.fit_transform(cnv_feat)
        pw_norm = scaler_path.fit_transform(pw_feat)

        ge_tensor = torch.tensor(ge_norm, device=device)
        cnv_tensor = torch.tensor(cnv_norm, device=device)
        pw_tensor = torch.tensor(pw_norm, device=device)

        # SDAE dimension reduction
        sdae_ge = train_sdae(ge_tensor, device, lr=cfg.sdae_lr, epochs=cfg.sdae_epochs, batch_size=cfg.sdae_batch)
        with torch.no_grad():
            exp_enc = sdae_ge(ge_tensor).cpu().numpy()
        np.save(exp_path, exp_enc)
        torch.save(sdae_ge.state_dict(), os.path.join(data_root, 'sdae_ge.pt'))

        sdae_cnv = train_sdae(cnv_tensor, device, lr=cfg.sdae_lr, epochs=cfg.sdae_epochs, batch_size=cfg.sdae_batch)
        with torch.no_grad():
            cnv_enc = sdae_cnv(cnv_tensor).cpu().numpy()
        np.save(cnv_path, cnv_enc)
        torch.save(sdae_cnv.state_dict(), os.path.join(data_root, 'sdae_cnv.pt'))

        np.save(path_path, pw_norm)
        joblib.dump(scaler_path, os.path.join(data_root, 'path_scaler.pkl'))

        with open(ach_list_path, 'w') as f:
            for ach in valid_achs:
                f.write(ach + '\n')
        print(f"Saved valid ACH IDs to {ach_list_path}")

        return np.load(exp_path), np.load(cnv_path), np.load(path_path), valid_achs
    else:
        if os.path.exists(ach_list_path):
            with open(ach_list_path, 'r') as f:
                valid_achs = [line.strip() for line in f if line.strip()]
            print(f"Loaded valid ACH IDs from {ach_list_path}")
        else:
            valid_achs = ach_ids
        return np.load(exp_path), np.load(cnv_path), np.load(path_path), valid_achs


# ==================== Drug graph construction ====================
def build_drug_graphs(drug_name_to_smiles, data_root):
    cache = os.path.join(data_root, 'drug_graphs.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    graphs = {}
    for name, smiles in drug_name_to_smiles.items():
        try:
            graphs[name] = smile_to_graph(smiles)
        except Exception as e:
            print(f"Failed for {name}: {e}")
    with open(cache, 'wb') as f:
        pickle.dump(graphs, f)
    return graphs


# ==================== Dataset ====================
class DrugPairDataset(Dataset):
    def __init__(self, pairs, drug_name_to_idx):
        self.samples = [(c, drug_name_to_idx[a], drug_name_to_idx[b], l)
                        for c, a, b, l in pairs
                        if a in drug_name_to_idx and b in drug_name_to_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        c, da, db, l = self.samples[idx]
        return (c, da, db, l)


def collate_fn(batch):
    cell_idx, da_idx, db_idx, labels = zip(*batch)
    return (torch.tensor(cell_idx),
            torch.tensor(da_idx),
            torch.tensor(db_idx),
            torch.tensor(labels, dtype=torch.float32))