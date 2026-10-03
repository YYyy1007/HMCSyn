import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATConv, GlobalAttention


def learnable_incidence_hg(x, tau, mask=None):
    """
    Learnable hypergraph incidence matrix.
    If mask is provided, only training nodes are used to compute similarities.
    """
    N = x.shape[0]
    if mask is not None:
        idx_train = mask.nonzero(as_tuple=True)[0]
        if len(idx_train) == 0:
            return torch.zeros(N, N, device=x.device)
        x_sub = x[idx_train]
        x_norm = F.normalize(x_sub, dim=1)
        S = x_norm @ x_norm.T
        probs = F.softmax(S / tau, dim=1)
        H = torch.zeros(N, N, device=x.device)
        H[idx_train[:, None], idx_train] = probs
        return H
    else:
        x_norm = F.normalize(x, dim=1)
        S = x_norm @ x_norm.T
        probs = F.softmax(S / tau, dim=1)
        return probs


def fuse_hypergraphs(H1, H2, H3):
    """
    Fuse three hypergraph incidence matrices by averaging overlapping entries.
    """
    mask1, mask2, mask3 = H1 != 0, H2 != 0, H3 != 0
    Hf = torch.zeros_like(H1)
    common = mask1 & mask2 & mask3
    Hf[common] = (H1[common] + H2[common] + H3[common]) / 3.0
    m12 = mask1 & mask2 & ~mask3
    Hf[m12] = (H1[m12] + H2[m12]) / 2.0
    m13 = mask1 & ~mask2 & mask3
    Hf[m13] = (H1[m13] + H3[m13]) / 2.0
    m23 = ~mask1 & mask2 & mask3
    Hf[m23] = (H2[m23] + H3[m23]) / 2.0
    Hf[mask1 & ~mask2 & ~mask3] = H1[mask1 & ~mask2 & ~mask3]
    Hf[~mask1 & mask2 & ~mask3] = H2[~mask1 & mask2 & ~mask3]
    Hf[~mask1 & ~mask2 & mask3] = H3[~mask1 & ~mask2 & mask3]
    return Hf


class HypergraphConvAttention(nn.Module):
    """Hypergraph convolution with attention on hyperedges."""
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(out_dim, 1)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x, H):
        out = self.proj(x)
        edge_feat = H.T @ out
        edge_weights = torch.sigmoid(self.attn(edge_feat))
        weighted_edge = edge_weights * edge_feat
        out = H @ weighted_edge
        residual = self.residual_proj(x)
        out = self.norm(residual + self.dropout(out))
        return out


class CrossModalDynamicFusion(nn.Module):
    """Cross-modal fusion via multi-head attention."""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z_list):
        new_Z = []
        for i, z_q in enumerate(Z_list):
            kv = torch.cat([z for j, z in enumerate(Z_list) if j != i], dim=0)
            q = z_q.unsqueeze(0)
            kv = kv.unsqueeze(0)
            attn_out, _ = self.attn(query=q, key=kv, value=kv)
            attn_out = attn_out.squeeze(0)
            out = self.norm(z_q + self.dropout(attn_out))
            out = out + self.mlp(out)
            new_Z.append(out)
        return new_Z


class DrugEncoder(nn.Module):
    """Molecular graph encoder for drugs."""
    def __init__(self, node_dim, edge_dim, hidden=128):
        super().__init__()
        self.atom_embed = nn.Linear(node_dim, hidden)
        self.trans1 = TransformerConv(hidden, hidden, heads=4, edge_dim=edge_dim, dropout=0.1, concat=True)
        self.trans2 = TransformerConv(hidden * 4, hidden, heads=4, edge_dim=edge_dim, dropout=0.1, concat=True)
        self.gat = GATConv(hidden * 4, hidden, heads=1, dropout=0.1)
        self.global_attn = GlobalAttention(gate_nn=nn.Linear(hidden, 1))
        self.proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.atom_embed(x)
        x = self.trans1(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.trans2(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.gat(x, edge_index)
        x = F.gelu(x)
        x = self.global_attn(x, batch)
        return self.proj(x)


class SynergyPredictor(nn.Module):
    """Predict drug synergy from cell and drug embeddings."""
    def __init__(self, dim, num_heads=4, dropout=0.2, drug_dim=128):
        super().__init__()
        self.drug_proj = nn.Linear(drug_dim, dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                                dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.classifier = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))
        self.regressor = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, cell_emb, drug_a_emb, drug_b_emb):
        drug_a_emb = self.drug_proj(drug_a_emb)
        drug_b_emb = self.drug_proj(drug_b_emb)
        dv = torch.stack([drug_a_emb, drug_b_emb], dim=1)
        q = cell_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(query=q, key=dv, value=dv)
        attn_out = attn_out.squeeze(1)
        inter = self.norm(cell_emb + attn_out)
        inter = inter + self.mlp(inter)
        combined = torch.cat([inter, cell_emb], dim=1)
        cls_logit = self.classifier(combined)
        reg_pred = self.regressor(combined).squeeze()
        cls_prob = torch.sigmoid(cls_logit).squeeze()
        cls_prob = torch.clamp(cls_prob, min=1e-7, max=1 - 1e-7)
        return cls_prob, reg_pred


class HMCSyn(nn.Module):
    """
    HMCSyn: Multimodal Learnable Hypergraph-based Framework for Drug Synergy Prediction.
    Only the default variant (learnable hypergraph + hypergraph fusion) is implemented.
    """
    def __init__(self, cfg, exp_raw_dim, cnv_raw_dim, pathway_dim, drug_node_dim, drug_edge_dim, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.use_gating = cfg.use_gating

        self.exp_proj = nn.Linear(exp_raw_dim, cfg.cell_dim)
        self.cnv_proj = nn.Linear(cnv_raw_dim, cfg.cell_dim)
        self.path_proj = nn.Sequential(
            nn.Linear(pathway_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

        self.enc_exp = HypergraphConvAttention(cfg.cell_dim, cfg.hidden, cfg.hg_dropout)
        self.enc_cnv = HypergraphConvAttention(cfg.cell_dim, cfg.hidden, cfg.hg_dropout)
        self.enc_path = HypergraphConvAttention(cfg.cell_dim, cfg.hidden, cfg.hg_dropout)

        self.tau_exp = nn.Parameter(torch.ones(1))
        self.tau_cnv = nn.Parameter(torch.ones(1))
        self.tau_path = nn.Parameter(torch.ones(1))

        if self.use_gating:
            self.gate_exp = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.Sigmoid())
            self.gate_cnv = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.Sigmoid())
            self.gate_path = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.Sigmoid())

        self.fusion = CrossModalDynamicFusion(cfg.hidden, cfg.num_heads, cfg.hg_dropout)
        self.view_weights = nn.Parameter(torch.ones(3))
        self.final_hg = HypergraphConvAttention(cfg.hidden, cfg.hidden, cfg.hg_dropout)

        self.decoder_exp = nn.Linear(cfg.hidden, cfg.cell_dim)
        self.decoder_cnv = nn.Linear(cfg.hidden, cfg.cell_dim)
        self.decoder_path = nn.Linear(cfg.hidden, 256)

        self.drug_enc = DrugEncoder(drug_node_dim, drug_edge_dim, cfg.drug_dim)
        self.predictor = SynergyPredictor(cfg.hidden, cfg.num_heads, drug_dim=cfg.drug_dim)

    def forward_cell_branch(self, x_exp_raw, x_cnv_raw, x_path_raw, train_mask=None):
        z_exp = self.exp_proj(x_exp_raw)
        z_cnv = self.cnv_proj(x_cnv_raw)
        z_path = self.path_proj(x_path_raw)

        # Learnable hypergraph incidence matrices for each modality
        H_exp = learnable_incidence_hg(z_exp, self.tau_exp, mask=train_mask)
        H_cnv = learnable_incidence_hg(z_cnv, self.tau_cnv, mask=train_mask)
        H_path = learnable_incidence_hg(z_path, self.tau_path, mask=train_mask)

        Z_exp = self.enc_exp(z_exp, H_exp)
        Z_cnv = self.enc_cnv(z_cnv, H_cnv)
        Z_path = self.enc_path(z_path, H_path)

        if self.use_gating:
            Z_exp = Z_exp * self.gate_exp(Z_exp)
            Z_cnv = Z_cnv * self.gate_cnv(Z_cnv)
            Z_path = Z_path * self.gate_path(Z_path)

        # Fuse the three hypergraphs
        H_fused = fuse_hypergraphs(H_exp, H_cnv, H_path)

        Z_upd = self.fusion([Z_exp, Z_cnv, Z_path])
        w = torch.softmax(self.view_weights, dim=0)
        Z_fused = w[0] * Z_upd[0] + w[1] * Z_upd[1] + w[2] * Z_upd[2]

        C = self.final_hg(Z_fused, H_fused)

        recon_exp = self.decoder_exp(C)
        recon_cnv = self.decoder_cnv(C)
        recon_path = self.decoder_path(C)
        return Z_exp, Z_cnv, Z_path, C, recon_exp, recon_cnv, recon_path, z_exp, z_cnv, z_path

    def forward_drug(self, batch_graph):
        return self.drug_enc(batch_graph)

    def forward_pair(self, cell_emb, drug_a_emb, drug_b_emb):
        return self.predictor(cell_emb, drug_a_emb, drug_b_emb)

    @staticmethod
    def contrastive_loss(views, temperature=0.2):
        views_clean = []
        for v in views:
            v = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
            if torch.norm(v) == 0:
                v = torch.randn_like(v) * 0.01
            views_clean.append(v)
        loss = 0.0
        n = len(views_clean)
        for i in range(n):
            for j in range(i + 1, n):
                u = F.normalize(views_clean[i], dim=1)
                v = F.normalize(views_clean[j], dim=1)
                logits = u @ v.T / temperature
                labels = torch.arange(u.size(0), device=u.device)
                loss += (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5
        return loss / (n * (n - 1) / 2)