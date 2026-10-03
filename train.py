import os, sys, random, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.data import Batch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    average_precision_score,
    f1_score
)

from config import get_config
from model import HMCSyn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        focal = bce * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = alpha_t * focal
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_name(name):
    return str(name).strip().upper()


def load_drug_smiles_from_oneil(data_root):
    oneil_file = os.path.join(data_root, 'oneil_with_smiles.csv')
    if not os.path.exists(oneil_file):
        raise FileNotFoundError(f"Missing {oneil_file}")
    df = pd.read_csv(oneil_file, dtype=str)
    drug_map = {}
    for idx, row in df.iterrows():
        name1 = normalize_name(row['name1'])
        name2 = normalize_name(row['name2'])
        smi1 = str(row['drug1']).strip()
        smi2 = str(row['drug2']).strip()
        drug_map[name1] = smi1
        drug_map[name2] = smi2
    print(f"Loaded {len(drug_map)} unique drugs from oneil_with_smiles.csv")
    return drug_map


def find_best_threshold(y_true, y_prob, metric='f1'):
    best_th = 0.5
    best_score = 0.0
    for th in np.arange(0.3, 0.8, 0.02):
        preds = (y_prob > th).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, preds)
        elif metric == 'kappa':
            score = cohen_kappa_score(y_true, preds)
        else:
            score = accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_th = th
    return best_th, best_score


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def main():
    cfg = get_config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 1. Load oneilnew.csv
    new_file = os.path.join(cfg.data_root, 'oneilnew.csv')
    if not os.path.exists(new_file):
        raise FileNotFoundError("Missing oneilnew.csv")
    df = pd.read_csv(new_file, dtype=str)
    print(f"Loaded oneilnew.csv with {len(df)} rows")

    # Extract all unique ModelIDs
    all_achs = df['ModelID'].unique().tolist()
    print(f"Total unique ModelIDs in file: {len(all_achs)}")

    # 2. Generate or load features (by ACH-ID)
    from data import get_or_train_sdae_by_ach, build_drug_graphs, DrugPairDataset, collate_fn
    exp_feat, cnv_feat, path_feat, valid_achs = get_or_train_sdae_by_ach(cfg.data_root, all_achs, device, cfg)
    print(f"\nFinal features: Exp {exp_feat.shape}, CNV {cnv_feat.shape}, Pathway {path_feat.shape}")
    x_exp = torch.tensor(exp_feat, dtype=torch.float32, device=device)
    x_cnv = torch.tensor(cnv_feat, dtype=torch.float32, device=device)
    x_path = torch.tensor(path_feat, dtype=torch.float32, device=device)

    # Build ACH -> index mapping
    ach_to_idx = {ach: i for i, ach in enumerate(valid_achs)}
    num_cells = len(valid_achs)
    print(f"Number of valid cells: {num_cells}")

    # 3. Drug mapping (extract drug1, drug2 from oneilnew.csv)
    drug_map = {}
    for idx, row in df.iterrows():
        name1 = normalize_name(row['name1'])
        name2 = normalize_name(row['name2'])
        smi1 = str(row['drug1']).strip()
        smi2 = str(row['drug2']).strip()
        drug_map[name1] = smi1
        drug_map[name2] = smi2
    print(f"Loaded {len(drug_map)} unique drugs from oneilnew.csv")

    # Build drug graphs (cached)
    drug_graphs = build_drug_graphs(drug_map, cfg.data_root)
    drug_names_list = list(drug_graphs.keys())
    drug_to_idx = {name: i for i, name in enumerate(drug_names_list)}
    print(f"Drug graphs built for {len(drug_to_idx)} drugs")

    # 4. Build sample pairs (directly use ModelID)
    pairs = []
    missing_drugs = set()
    skipped_achs = set()
    drug_keys = set(drug_to_idx.keys())

    for _, row in df.iterrows():
        ach = str(row['ModelID']).strip()
        drugA = normalize_name(row['name1'])
        drugB = normalize_name(row['name2'])
        label = int(row['label'])

        if ach not in ach_to_idx:
            skipped_achs.add(ach)
            continue
        if drugA not in drug_keys or drugB not in drug_keys:
            if drugA not in drug_keys:
                missing_drugs.add(drugA)
            if drugB not in drug_keys:
                missing_drugs.add(drugB)
            continue
        cidx = ach_to_idx[ach]
        pairs.append((cidx, drugA, drugB, label))

    # Statistics
    total_original = len(df)
    valid_count = len(pairs)
    synergy_count = sum(1 for p in pairs if p[3] == 1)
    antag_count = sum(1 for p in pairs if p[3] == 0)
    unique_drugs = set()
    for p in pairs:
        unique_drugs.add(p[1])
        unique_drugs.add(p[2])
    unique_cells = set(p[0] for p in pairs)

    print("\n" + "=" * 60)
    print("Dataset Statistics After Preprocessing:")
    print(f"  Total triples in file                 : {total_original}")
    print(f"  Valid triples after filtering          : {valid_count}")
    print(f"  Synergistic pairs (label=1)            : {synergy_count}")
    print(f"  Antagonistic pairs (label=0)           : {antag_count}")
    print(f"  Unique drugs involved                  : {len(unique_drugs)}")
    print(f"  Unique cells involved                  : {len(unique_cells)}")
    if skipped_achs:
        print(f"  ACHs not matched (examples)          : {list(skipped_achs)[:5]}")
    if missing_drugs:
        print(f"  Drugs not found in graph (examples)  : {list(missing_drugs)[:5]}")
    print("=" * 60 + "\n")

    if len(pairs) == 0:
        raise RuntimeError("No valid pairs!")

    # ========== 5. Stratified cross-validation ==========
    labels = np.array([p[3] for p in pairs])
    skf = StratifiedKFold(n_splits=cfg.num_splits, shuffle=True, random_state=cfg.seed)
    all_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(pairs, labels)):
        print(f"\n{'=' * 40}\nFold {fold_idx + 1}/{cfg.num_splits}")
        train_pairs = [pairs[i] for i in train_idx]
        val_pairs = [pairs[i] for i in val_idx]

        train_indices = set()
        for idx, _, _, _ in train_pairs:
            train_indices.add(idx)
        train_mask = torch.zeros(num_cells, dtype=torch.bool, device=device)
        for idx in train_indices:
            train_mask[idx] = True

        train_ds = DrugPairDataset(train_pairs, drug_to_idx)
        val_ds = DrugPairDataset(val_pairs, drug_to_idx)
        if len(train_ds) == 0 or len(val_ds) == 0:
            continue

        train_labels = np.array([s[3] for s in train_ds.samples])
        pos = np.sum(train_labels == 1)
        neg = len(train_labels) - pos

        if pos == 0 or neg == 0:
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        else:
            pos_w = len(train_labels) / (2 * pos)
            neg_w = len(train_labels) / (2 * neg)
            weights = torch.tensor(np.where(train_labels == 1, pos_w, neg_w), dtype=torch.float32)
            sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, collate_fn=collate_fn)

        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

        sample_graph = drug_graphs[drug_names_list[0]]
        model = HMCSyn(cfg,
                       exp_feat.shape[1],
                       cnv_feat.shape[1],
                       path_feat.shape[1],
                       sample_graph.x.size(1),
                       sample_graph.edge_attr.size(1),
                       device).to(device)

        criterion_cls = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, smoothing=cfg.label_smoothing)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        total_steps = len(train_loader) * cfg.epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, total_steps=total_steps,
            pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=1000.0
        )

        ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

        best_auc = 0.0
        no_improve = 0
        best_model_state = None
        best_ema_state = None

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            loss_epoch = 0.0
            for cell_idx, da_idx, db_idx, labels in train_loader:
                cell_idx, labels = cell_idx.to(device), labels.to(device)
                Z_exp, Z_cnv, Z_path, C, recon_exp, recon_cnv, recon_path, z_exp, z_cnv, z_path = model.forward_cell_branch(
                    x_exp, x_cnv, x_path, train_mask=train_mask
                )

                if torch.isnan(Z_exp).any() or torch.isnan(Z_cnv).any() or torch.isnan(Z_path).any() or torch.isnan(C).any():
                    print(f"Warning: NaN detected in cell representations at epoch {epoch}. Skipping batch.")
                    continue

                drug_a_list = [drug_graphs[drug_names_list[i]] for i in da_idx]
                drug_b_list = [drug_graphs[drug_names_list[i]] for i in db_idx]
                batch_a = Batch.from_data_list(drug_a_list).to(device)
                batch_b = Batch.from_data_list(drug_b_list).to(device)

                d_a = model.forward_drug(batch_a)
                d_b = model.forward_drug(batch_b)
                cls_prob, _ = model.forward_pair(C[cell_idx], d_a, d_b)

                cls_prob = torch.nan_to_num(cls_prob, nan=0.5)
                cls_prob = torch.clamp(cls_prob, min=1e-7, max=1 - 1e-7)

                loss_cls = criterion_cls(cls_prob, labels)
                loss_con = model.contrastive_loss([Z_exp, Z_cnv, Z_path, C], cfg.temperature) * cfg.contrastive_weight
                loss_recon = (F.mse_loss(recon_exp, z_exp) +
                              F.mse_loss(recon_cnv, z_cnv) +
                              F.mse_loss(recon_path, z_path)) / 3.0 if cfg.recon_weight > 0 else torch.tensor(0.0, device=device)

                loss = loss_cls + loss_con + cfg.recon_weight * loss_recon

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update()
                loss_epoch += loss.item()

            # Validation
            model.eval()
            if ema is not None:
                ema.apply_shadow()

            y_true_cls, y_prob = [], []
            with torch.no_grad():
                Z_exp, Z_cnv, Z_path, C, _, _, _, _, _, _ = model.forward_cell_branch(
                    x_exp, x_cnv, x_path, train_mask=train_mask
                )
                for cell_idx, da_idx, db_idx, labels in val_loader:
                    cell_idx, labels = cell_idx.to(device), labels.to(device)
                    drug_a_list = [drug_graphs[drug_names_list[i]] for i in da_idx]
                    drug_b_list = [drug_graphs[drug_names_list[i]] for i in db_idx]
                    batch_a = Batch.from_data_list(drug_a_list).to(device)
                    batch_b = Batch.from_data_list(drug_b_list).to(device)
                    d_a = model.forward_drug(batch_a)
                    d_b = model.forward_drug(batch_b)
                    cls_prob, _ = model.forward_pair(C[cell_idx], d_a, d_b)
                    y_true_cls.append(labels.cpu())
                    y_prob.append(cls_prob.cpu())

            if ema is not None:
                ema.restore()

            y_true_cls = torch.cat(y_true_cls).numpy()
            y_prob = torch.cat(y_prob).numpy()

            if len(np.unique(y_true_cls)) < 2:
                auc = float('nan')
            else:
                auc = roc_auc_score(y_true_cls, y_prob)

            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if ema is not None:
                    best_ema_state = copy.deepcopy(ema.shadow)
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 50 == 0:
                print(f'Epoch {epoch}: train loss={loss_epoch / len(train_loader):.4f}, val AUC={auc:.4f}')

            if no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_ema_state is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data = best_ema_state[name]
            print("Loaded best EMA model.")
        elif best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(device)
            print("Loaded best regular model.")
        else:
            print(f"Fold {fold_idx + 1}: No improvement, using last model.")

        # Final validation
        model.eval()
        y_true_cls, y_prob = [], []
        with torch.no_grad():
            Z_exp, Z_cnv, Z_path, C, _, _, _, _, _, _ = model.forward_cell_branch(
                x_exp, x_cnv, x_path, train_mask=train_mask
            )
            for cell_idx, da_idx, db_idx, labels in val_loader:
                cell_idx, labels = cell_idx.to(device), labels.to(device)
                drug_a_list = [drug_graphs[drug_names_list[i]] for i in da_idx]
                drug_b_list = [drug_graphs[drug_names_list[i]] for i in db_idx]
                batch_a = Batch.from_data_list(drug_a_list).to(device)
                batch_b = Batch.from_data_list(drug_b_list).to(device)
                d_a = model.forward_drug(batch_a)
                d_b = model.forward_drug(batch_b)
                cls_prob, _ = model.forward_pair(C[cell_idx], d_a, d_b)
                y_true_cls.append(labels.cpu())
                y_prob.append(cls_prob.cpu())

        y_true_cls = torch.cat(y_true_cls).numpy()
        y_prob = torch.cat(y_prob).numpy()

        if len(np.unique(y_true_cls)) < 2:
            print(f"Fold {fold_idx + 1}: validation set has only one class, skipping metrics.")
            all_metrics.append(None)
            continue

        best_th, best_f1 = find_best_threshold(y_true_cls, y_prob, metric='f1')
        preds = (y_prob > best_th).astype(int)
        acc = accuracy_score(y_true_cls, preds)
        precision = precision_score(y_true_cls, preds, zero_division=0)
        recall = recall_score(y_true_cls, preds, zero_division=0)
        kappa = cohen_kappa_score(y_true_cls, preds)
        auc = roc_auc_score(y_true_cls, y_prob)
        pr_auc = average_precision_score(y_true_cls, y_prob)

        fold_metrics = {
            'ACC': acc,
            'Precision': precision,
            'Recall': recall,
            'Kappa': kappa,
            'ROC_AUC': auc,
            'PR_AUC': pr_auc,
            'Best_Threshold': best_th
        }
        print(f"Fold {fold_idx + 1} best (threshold={best_th:.2f}): "
              f"ACC={acc:.4f}, Prec={precision:.4f}, Recall={recall:.4f}, "
              f"Kappa={kappa:.4f}, ROC_AUC={auc:.4f}, PR_AUC={pr_auc:.4f}")
        all_metrics.append(fold_metrics)

    valid_metrics = [m for m in all_metrics if m is not None]
    if valid_metrics:
        avg = {key: np.nanmean([m[key] for m in valid_metrics]) for key in valid_metrics[0] if key != 'Best_Threshold'}
        std = {key: np.nanstd([m[key] for m in valid_metrics]) for key in valid_metrics[0] if key != 'Best_Threshold'}
        print("\n" + "=" * 60)
        print(f"Final Average Metrics over {len(valid_metrics)} folds:")
        print(f"ACC       = {avg['ACC']:.4f} ± {std['ACC']:.4f}")
        print(f"Precision = {avg['Precision']:.4f} ± {std['Precision']:.4f}")
        print(f"Recall    = {avg['Recall']:.4f} ± {std['Recall']:.4f}")
        print(f"Kappa     = {avg['Kappa']:.4f} ± {std['Kappa']:.4f}")
        print(f"ROC AUC   = {avg['ROC_AUC']:.4f} ± {std['ROC_AUC']:.4f}")
        print(f"PR AUC    = {avg['PR_AUC']:.4f} ± {std['PR_AUC']:.4f}")
        avg_th = np.mean([m['Best_Threshold'] for m in valid_metrics])
        print(f"Average best threshold = {avg_th:.3f}")
    else:
        print("\nNo valid folds with both classes for metric calculation.")


if __name__ == "__main__":
    main()