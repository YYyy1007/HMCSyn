import pandas as pd
import csv
import re
import os

from config import get_config


def clean_name_for_match(name):
    """Clean name for matching: remove non-alphanumeric characters and convert to uppercase."""
    return re.sub(r'[^A-Z0-9]', '', str(name).upper())


def extract_base_cand_name(name):
    """Extract prefix of feature library name (remove _tissue name)."""
    name = str(name).upper()
    base = name.split('_')[0]
    # Handle X prefix exported by R
    if re.match(r'^X\d', base):
        base = base[1:]
    return re.sub(r'[^A-Z0-9]', '', base), name


def strict_match(target, candidates):
    target_clean = clean_name_for_match(target)

    # --- Safe alias whitelist ---
    safe_alias_dict = {
        "LNCAP": "LNCAPCLONEFGC",
        "OVCAR3": "NIHOVCAR3",
        "MSTO": "MSTO211H"
    }

    check_target = safe_alias_dict.get(target_clean, target_clean)

    # --- Unique priority: exact match of base name ---
    for cand in candidates:
        cand_base, _ = extract_base_cand_name(cand)
        if check_target == cand_base:
            return cand, "Exact (exact match or safe alias)"

    return None, "Not Found"


def execute_strict_pipeline(dataset_file=None, features_file=None):
    if dataset_file is None or features_file is None:
        cfg = get_config()
        data_root = cfg.data_root
        if dataset_file is None:
            dataset_file = os.path.join(data_root, 'oneil_with_smiles.csv')
        if features_file is None:
            features_file = os.path.join(data_root, 'cell_features.csv')

    print("================ Start strict alignment and cleaning pipeline ================")

    # --- 1. Read data ---
    try:
        df_dataset = pd.read_csv(dataset_file)
    except Exception as e:
        print(f"Failed to read dataset: {e}")
        return

    try:
        with open(features_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            feature_rows = list(reader)
        header_1 = feature_rows[0]
        header_2 = feature_rows[1]
        feature_cells_dict = {row[0]: row for row in feature_rows[2:] if len(row) > 0}
        candidates = list(feature_cells_dict.keys())
    except Exception as e:
        print(f"Failed to read feature library: {e}")
        return

    # --- 2. Strict cell line matching ---
    if 'cell_line' not in df_dataset.columns and 'cell' in df_dataset.columns:
        df_dataset.rename(columns={'cell': 'cell_line'}, inplace=True)
    original_cells = df_dataset['cell_line'].unique().tolist()
    valid_mapping = {}
    missing_cells = []
    match_logs = []

    print("Performing strict cell line matching...")
    for target in original_cells:
        best_cand, match_type = strict_match(target, candidates)
        if best_cand:
            valid_mapping[target] = best_cand
            match_logs.append(f"[{match_type}] {target} -> {best_cand}")
        else:
            missing_cells.append(target)

    # --- 3. Remove missing samples ---
    original_sample_count = len(df_dataset)
    df_strict = df_dataset[df_dataset['cell_line'].isin(valid_mapping.keys())].copy()
    new_sample_count = len(df_strict)
    dropped_count = original_sample_count - new_sample_count

    # --- 4. Statistics ---
    if 'drug_a_name' not in df_strict.columns and 'name1' in df_strict.columns:
        df_strict.rename(columns={'name1': 'drug_a_name', 'name2': 'drug_b_name'}, inplace=True)
    unique_drugs = pd.concat([df_strict['drug_a_name'], df_strict['drug_b_name']]).unique()
    unique_cells = df_strict['cell_line'].unique()

    label_counts = df_strict['label'].value_counts()
    pos_count = label_counts.get(1, 0)
    neg_count = label_counts.get(0, 0)
    pos_ratio = pos_count / new_sample_count * 100 if new_sample_count > 0 else 0
    neg_ratio = neg_count / new_sample_count * 100 if new_sample_count > 0 else 0

    # --- 5. Extract features ---
    final_features = []
    for cell in unique_cells:
        mapped_name = valid_mapping[cell]
        final_features.append(feature_cells_dict[mapped_name])

    # --- 6. Export files ---
    print("Generating final files...")

    df_strict.to_csv('dataset_S10.csv', index=False)

    pd.Series(unique_drugs).dropna().sort_values().to_csv('final_unique_drugs.txt', index=False, header=False)
    pd.Series(unique_cells).dropna().sort_values().to_csv('final_unique_cell_lines.txt', index=False, header=False)

    with open('final_matched_cell_features.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header_1)
        writer.writerow(header_2)
        writer.writerows(final_features)

    # --- 7. Report ---
    report_text = (
            "==================================================\n"
            "           Final strict cleaning and alignment report             \n"
            "==================================================\n"
            f"1. Cell line removal results:\n"
            f"   - Original data contains {len(original_cells)} cell lines.\n"
            f"   - Removed {len(missing_cells)} cell lines due to failed exact matching: {', '.join(missing_cells)}\n"
            f"2. Sample cleaning results:\n"
            f"   - Original sample count: {original_sample_count}\n"
            f"   - Removed unmatched samples: {dropped_count}\n"
            f"   - Final retained samples: {new_sample_count}\n"
            f"3. Final label distribution:\n"
            f"   - Class 1 (Synergy > 10): {pos_count} samples ({pos_ratio:.2f}%)\n"
            f"   - Class 0 (Synergy <= 10): {neg_count} samples ({neg_ratio:.2f}%)\n"
            f"4. Final unique drug types: {len(unique_drugs)}\n"
            f"5. Final unique cell line types: {len(unique_cells)}\n"
            "==================================================\n"
            "--- Details of successfully aligned cell lines ---\n" +
            "\n".join(sorted(match_logs)) + "\n"
    )

    with open('final_dataset_statistics_S10.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print("\nTask completed! The following files have been generated:")
    print(" - dataset_S10.csv (final strict dataset)")
    print(" - final_matched_cell_features.csv (aligned feature set)")
    print(" - final_dataset_statistics_S10.txt (latest statistics report)")
    print(" - final_unique_drugs.txt")
    print(" - final_unique_cell_lines.txt")


if __name__ == '__main__':
    execute_strict_pipeline()