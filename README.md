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

Place the following files under `data/`:

- `oneilnew.csv` — drug-pair dataset with columns: `ModelID`, `name1`, `name2`, `drug1`, `drug2`, `label`
- `Model.csv` — cell-line metadata with `ModelID`, `CellLineName`
- `OmicsExpressionTPMLogp1HumanProteinCoding.csv` — gene expression
- `OmicsCNGeneWGS.csv` — copy number variation
- `ONeil_31_gsva_1329_dim.csv` — pathway activity


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

All intermediate features (SDAE-reduced matrices, drug graphs, valid ACH list) are generated automatically on the first run.

## Code Overview

- `train.py` — main training and evaluation
- `model.py` — HMCSyn model 
- `data.py` — data loading, SDAE feature reduction, dataset classes
- `config.py` — hyperparameters
- `process.py` — strict cell-line matching and cleaning 
- `build_drug_graphs.py` — build drug molecular graphs
