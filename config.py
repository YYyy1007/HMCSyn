import argparse


def get_config():
    parser = argparse.ArgumentParser(description="HMCSyn - Enhanced for AUC")
    parser.add_argument("--data_root", type=str, default="./data", help="Path to ONeil data folder")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    # SDAE
    parser.add_argument("--sdae_lr", type=float, default=1e-4)
    parser.add_argument("--sdae_epochs", type=int, default=2500)
    parser.add_argument("--sdae_batch", type=int, default=30)

    # HMCSyn hyperparameters
    parser.add_argument("--cell_dim", type=int, default=256, help="Projection dim for each modality")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dim")
    parser.add_argument("--drug_dim", type=int, default=128)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--hg_dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--contrastive_weight", type=float, default=0.5, help="Contrastive loss weight")
    parser.add_argument("--temperature", type=float, default=0.2)

    # Loss settings
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--focal_alpha", type=float, default=0.3)
    parser.add_argument("--reg_weight", type=float, default=2.0)
    parser.add_argument("--recon_weight", type=float, default=0.02)
    parser.add_argument("--use_huber_loss", action="store_true", default=False)

    # Label smoothing
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    parser.add_argument("--use_gating", action="store_true", default=True)
    parser.add_argument("--use_fp", action="store_true", default=True)
    parser.add_argument("--fp_dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr_scheduler_patience", type=int, default=10)
    parser.add_argument("--lr_factor", type=float, default=0.75)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--num_splits", type=int, default=5)

    # EMA
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    return parser.parse_args()