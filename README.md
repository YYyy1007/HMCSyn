# HMCSyn: A Multimodal Learnable Hypergraph-based Framework for Drug Synergy Prediction

Official implementation of HMCSyn for drug synergy prediction.

## Requirements

- Python 3.8+
- PyTorch 1.12+ (CUDA recommended)
- torch-geometric, rdkit, numpy, pandas, scikit-learn, joblib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

### Cell lines

- `ONeil_31_gsva_1329_dim.csv` — Pathway activity scores for 30 cell lines (GSVA, 1329 pathways).
- `Model.csv` — Cell line metadata (`ModelID`, `CellLineName`).
- Gene expression data: downloaded from CCLE (Cancer Cell Line Encyclopedia). File: `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`.
- DNA copy number data: downloaded from CCLE (Cancer Cell Line Encyclopedia). File: `OmicsCNGeneWGS.csv`.

### Drugs

- SMILES strings for all drugs are provided in `oneilnew.csv` (columns `drug1` and `drug2`).
- Drug molecular graphs are generated automatically by `build_drug_graphs.py` or during training.

### Dataset

- `oneilnew.csv` — Drug pair–cell line triples. Each row contains:
  - `ModelID` — cell line ACH-ID
  - `name1`, `name2` — drug names
  - `drug1`, `drug2` — SMILES strings
  - `label` — synergy label (1 = synergistic, 0 = antagonistic)
- The dataset covers 538 pairwise drug combinations across 30 cell lines.

### Data access

The following files are included in this repository under `data/`:

- `oneilnew.csv`
- `Model.csv`
- `ONeil_31_gsva_1329_dim.csv`

The two large raw omics files are not included due to GitHub file size limits:

- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
- `OmicsCNGeneWGS.csv`

They are publicly available from CCLE (Cancer Cell Line Encyclopedia) and can be obtained via the DepMap portal (https://depmap.org). Place them in the `data/` folder with the exact file names above.

After all files are placed under `data/`, run:

```bash
python train.py --data_root ./data --device cuda:0 --seed 42
```

All intermediate features (SDAE-reduced matrices, drug graphs, valid ACH list) are generated automatically on the first run.

## Training

```bash
python train.py --data_root ./data --device cuda:0 --seed 42
```

The script performs stratified 5-fold cross-validation and reports ACC, Precision, Recall, Kappa, ROC AUC, PR AUC, and the best threshold.

Key arguments:

- `--data_root` : data directory (default `./data`)
- `--device` : `cuda:0` or `cpu`
- `--seed` : random seed (default 42)
- `--num_splits` : number of CV folds (default 5)
- `--epochs` : max epochs (default 500)

## Code Overview

- `train.py` — main training and evaluation
- `model.py` — HMCSyn model 
- `data.py` — data loading, SDAE feature reduction, dataset classes
- `config.py` — hyperparameters
- `process.py` — strict cell-line matching and cleaning
- `build_drug_graphs.py` — build drug molecular graphs


## Acknowledgements

The SDAE implementation is adapted from [GADRP](https://github.com/flora619/GADRP) with minor modifications.
