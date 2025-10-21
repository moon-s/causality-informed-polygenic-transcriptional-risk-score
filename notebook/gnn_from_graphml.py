"""
gnn_from_graphml.py

End-to-end utilities to:
  1) Load a NetworkX GraphML with node attributes (community, beta, etc.)
  2) Build a PyTorch Geometric `Data` object directly (no CSVs)
  3) Train a GCN/GAT regressor to impute per-gene MR betas (semi-supervised)
  4) Export predictions

Compatible with your existing notebook's model/trainer interface.
"""

from __future__ import annotations
import os
from typing import Dict, Tuple, Optional, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATv2Conv, BatchNorm
from torch_geometric.utils import add_self_loops


# -------------------------
# Reproducibility helpers
# -------------------------
def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Data builder
# -------------------------
def build_data_from_graphml(
    graphml_path: str,
    submodule: Optional[int] = None,
    add_degree_features: bool = True,
    use_label_as_feat: bool = True,
    add_community_feature: bool = False,
    community_max_id: Optional[int] = None,
    beta_attr: str = "beta",
    weight_attr: str = "weight",
    community_attr: str = "community",
) -> Tuple[Data, Dict[int, str], Dict[str, int]]:
    """
    Build a torch_geometric.data.Data from a GraphML file.

    Expected node attrs:
      - community (int) : submodule id (0..12)
      - beta (float)    : MR beta (NaN if unknown / unlabeled)

    Expected edge attr (optional):
      - weight (float)  : edge weight (defaults to 1.0 if missing)

    Args
    ----
    graphml_path : str
        Path to GraphML.
    submodule : Optional[int]
        If provided, filter nodes to this community id.
    add_degree_features : bool
        Include z-scored degree as a node feature.
    use_label_as_feat : bool
        Include prior beta (0 if unknown) and is_labeled as features.
    add_community_feature : bool
        If True, append a one-hot encoding of `community` as features
        (useful when training on the total module).
    community_max_id : Optional[int]
        Max community id for one-hot length. If None, inferred from graph.
    beta_attr, weight_attr, community_attr : str
        Attribute names in the GraphML.

    Returns
    -------
    data : torch_geometric.data.Data
        PyG Data with fields: x, y, edge_index, edge_weight, edge_attr, has_label, idx2gene
    idx2gene : Dict[int, str]
    gene2idx : Dict[str, int]
    """
    if not os.path.exists(graphml_path):
        raise FileNotFoundError(f"GraphML not found: {graphml_path}")

    G_full = nx.read_graphml(graphml_path)

    # Optional filter to one submodule
    if submodule is not None:
        keep = [n for n, d in G_full.nodes(data=True)
                if int(d.get(community_attr, -1)) == int(submodule)]
        G = G_full.subgraph(keep).copy()
    else:
        G = G_full

    # Safety: if empty, raise
    if G.number_of_nodes() == 0:
        raise ValueError("Subgraph has zero nodes (submodule filter too strict?).")

    nodes = list(G.nodes())
    gene2idx = {g: i for i, g in enumerate(nodes)}
    idx2gene = {i: g for g, i in gene2idx.items()}

    # Build edges (undirected)
    src, dst, ew = [], [], []
    for u, v, ed in G.edges(data=True):
        w = float(ed.get(weight_attr, 1.0))
        iu, iv = gene2idx[u], gene2idx[v]
        src += [iu, iv]
        dst += [iv, iu]
        ew += [w, w]

    # If isolated nodes: still allow graph with only self-loops
    if len(src) == 0:
        # create trivial self-loop to avoid zero-edge graph errors downstream
        src, dst, ew = [0], [0], [1.0]

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    ew = torch.tensor(np.array(ew, dtype=np.float32))
    edge_index, ew = add_self_loops(edge_index, edge_attr=ew, fill_value=1.0, num_nodes=len(nodes))

    # Labels from node attribute `beta_attr`
    y = torch.full((len(nodes), 1), float("nan"), dtype=torch.float32)
    has_label = torch.zeros(len(nodes), dtype=torch.bool)
    for g, i in gene2idx.items():
        b = G.nodes[g].get(beta_attr, None)
        if b is not None and not (isinstance(b, float) and np.isnan(b)):
            y[i, 0] = float(b)
            has_label[i] = True

    # Build node features
    feats: List[np.ndarray] = []

    if add_degree_features:
        deg = np.array([G.degree(g) for g in nodes], dtype=np.float32)
        deg = (deg - deg.mean()) / (deg.std() + 1e-8)
        feats.append(deg.reshape(-1, 1))

    if use_label_as_feat:
        prior = np.zeros((len(nodes), 1), dtype=np.float32)
        mask_np = has_label.numpy()
        if mask_np.any():
            prior[mask_np, 0] = y[mask_np, 0].numpy().reshape(-1)
        feats.append(prior)
        feats.append(mask_np.astype(np.float32).reshape(-1, 1))

    if add_community_feature:
        # one-hot encode community id
        comm_ids = np.array([int(G.nodes[g].get(community_attr, 0)) for g in nodes], dtype=int)
        if community_max_id is None:
            K = int(comm_ids.max()) + 1
        else:
            K = int(community_max_id) + 1
        oh = np.zeros((len(nodes), K), dtype=np.float32)
        oh[np.arange(len(nodes)), comm_ids] = 1.0
        feats.append(oh)

    if feats:
        x_np = np.hstack(feats)
    else:
        x_np = np.ones((len(nodes), 1), dtype=np.float32)

    x = torch.from_numpy(x_np).float()

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        edge_weight=ew,           # used by GCN
        edge_attr=ew.view(-1, 1), # used by GAT (edge_dim=1)
        has_label=has_label,
        idx2gene=idx2gene,
    )
    return data, idx2gene, gene2idx


# -------------------------
# Models
# -------------------------
class GCNReg(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dims = [in_dim] + [hidden] * (layers - 1)
        for i in range(layers - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1], normalize=True))
            self.bns.append(BatchNorm(dims[i + 1]))
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class GATReg(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        # first
        self.layers.append(
            GATv2Conv(
                in_channels=in_dim,
                out_channels=hidden,
                heads=heads,
                dropout=attn_dropout,
                concat=True,
                edge_dim=1,
            )
        )
        self.bns.append(BatchNorm(hidden * heads))
        # hidden
        for _ in range(layers - 2):
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden * heads,
                    out_channels=hidden,
                    heads=heads,
                    dropout=attn_dropout,
                    concat=True,
                    edge_dim=1,
                )
            )
            self.bns.append(BatchNorm(hidden * heads))
        # last to hidden
        if layers > 1:
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden * heads,
                    out_channels=hidden,
                    heads=1,
                    dropout=attn_dropout,
                    concat=False,
                    edge_dim=1,
                )
            )
            self.bns.append(BatchNorm(hidden))
            last_dim = hidden
        else:
            last_dim = hidden * heads
        self.head = nn.Linear(last_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        for conv, bn in zip(self.layers, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


def build_model(model_type: str, in_dim: int, hps: Dict) -> nn.Module:
    """Factory for GCN/GAT with hyperparameters dict."""
    mt = model_type.upper()
    if mt == "GCN":
        return GCNReg(
            in_dim=in_dim,
            hidden=hps.get("hidden", 128),
            layers=hps.get("layers", 3),
            dropout=hps.get("dropout", 0.2),
        )
    elif mt == "GAT":
        return GATReg(
            in_dim=in_dim,
            hidden=hps.get("hidden", 128),
            layers=hps.get("layers", 3),
            heads=hps.get("heads", 4),
            dropout=hps.get("dropout", 0.2),
            attn_dropout=hps.get("attn_dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(
        self,
        data: Data,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        patience: int = 50,
    ):
        self.data = data
        self.model = model.to(DEVICE)

        # tensors
        self.x = data.x.to(DEVICE).float()
        self.edge_index = data.edge_index.to(DEVICE)
        self.edge_weight = getattr(data, "edge_weight", None)
        if self.edge_weight is not None:
            self.edge_weight = self.edge_weight.to(DEVICE).float()
        self.edge_attr = getattr(data, "edge_attr", None)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(DEVICE).float()

        self.y = data.y.to(DEVICE).float()
        self.has_label = data.has_label.to(DEVICE)

        # optimizer & scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.patience = patience
        self.best_state, self.best_val, self.bad_epochs = None, float("inf"), 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # masks
        self.train_mask = None
        self.val_mask = None

    def split_masks(self, val_size: float = 0.2, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        labeled_idx = torch.nonzero(self.has_label, as_tuple=False).view(-1)
        n = labeled_idx.numel()
        if n < 2:
            # minimal safety: at least 1 labeled for train, 1 for val if possible
            raise ValueError(f"Not enough labeled nodes to split (found {n}).")
        perm = torch.randperm(n, device=labeled_idx.device)
        n_val = max(1, int(round(n * val_size)))
        n_val = min(n_val, n - 1)  # leave at least 1 for train
        val_idx = labeled_idx[perm[:n_val]]
        train_idx = labeled_idx[perm[n_val:]]
        train_mask = torch.zeros_like(self.has_label, dtype=torch.bool)
        val_mask = torch.zeros_like(self.has_label, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        self.train_mask, self.val_mask = train_mask, val_mask
        return train_mask, val_mask

    def _loss(self, mask: torch.Tensor) -> torch.Tensor:
        pred = self.model(self.x, self.edge_index, edge_weight=self.edge_weight, edge_attr=self.edge_attr)
        y_true = self.y[mask]
        y_pred = pred[mask]
        return F.mse_loss(y_pred, y_true)

    def train(self, epochs: int = 500, verbose: bool = True) -> None:
        if self.train_mask is None or self.val_mask is None:
            raise RuntimeError("Call split_masks() before train().")
        for ep in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            loss = self._loss(self.train_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # eval
            self.model.eval()
            with torch.no_grad():
                val_loss = self._loss(self.val_mask).item()
            tr_loss = float(loss.item())
            self.train_losses.append(tr_loss)
            self.val_losses.append(val_loss)

            improved = val_loss < self.best_val - 1e-9
            if improved:
                self.best_val = val_loss
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1

            if verbose and (ep % 25 == 0 or ep == 1):
                print(f"Epoch {ep:4d} | train {tr_loss:.6f} | val {val_loss:.6f}")

            if self.bad_epochs >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {ep} (best val={self.best_val:.6f})")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    @torch.no_grad()
    def predict_all(self) -> torch.Tensor:
        self.model.eval()
        pred = self.model(self.x, self.edge_index, edge_weight=self.edge_weight, edge_attr=self.edge_attr)
        return pred.squeeze(1).detach().cpu()

    def export_predictions(self, out_path: str) -> pd.DataFrame:
        pred_all = self.predict_all().numpy()
        df = pd.DataFrame({
            "gene": [self.data.idx2gene[i] for i in range(len(self.data.idx2gene))],
            "pred_beta": pred_all,
            "is_labeled": self.data.has_label.cpu().numpy().astype(bool),
        })
        # if true beta exists on those nodes, attach for convenience
        true_map = {}
        y_np = self.data.y.squeeze(1).cpu().numpy()
        has = self.data.has_label.cpu().numpy()
        for i, ok in enumerate(has):
            if ok:
                true_map[self.data.idx2gene[i]] = float(y_np[i])
        df["true_beta"] = df["gene"].map(true_map).astype(float)
        df.to_csv(out_path, sep="\t", index=False)
        return df


# -------------------------
# Convenience runner
# -------------------------
def train_from_graphml(
    graphml_path: str,
    outdir: str,
    submodule: Optional[int] = None,
    model_type: str = "GAT",
    seed: int = 42,
    # model hparams
    hidden: int = 128,
    layers: int = 3,
    heads: int = 4,
    dropout: float = 0.3,
    attn_dropout: float = 0.1,
    # trainer hparams
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    epochs: int = 600,
    patience: int = 80,
    val_size: float = 0.2,
    # features
    add_degree_features: bool = True,
    use_label_as_feat: bool = True,
    add_community_feature: bool = False,
    community_max_id: Optional[int] = None,
) -> Tuple[Trainer, pd.DataFrame]:
    """
    One-shot train & export on a (sub)module graph.

    Returns (trainer, predictions_df)
    """
    os.makedirs(outdir, exist_ok=True)
    set_seed(seed)

    data, idx2gene, gene2idx = build_data_from_graphml(
        graphml_path=graphml_path,
        submodule=submodule,
        add_degree_features=add_degree_features,
        use_label_as_feat=use_label_as_feat,
        add_community_feature=add_community_feature,
        community_max_id=community_max_id,
    )

    in_dim = data.x.shape[1]
    hps = dict(hidden=hidden, layers=layers, dropout=dropout, heads=heads, attn_dropout=attn_dropout)
    model = build_model(model_type, in_dim, hps)

    trainer = Trainer(
        data=data,
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )
    trainer.split_masks(val_size=val_size, seed=seed)
    trainer.train(epochs=epochs, verbose=True)

    tag = f"{model_type}_sub{('all' if submodule is None else submodule)}"
    out_path = os.path.join(outdir, f"{tag}_beta_predictions.tsv")
    pred_df = trainer.export_predictions(out_path)
    return trainer, pred_df


# -------------------------
# CLI usage example
# -------------------------
if __name__ == "__main__":
    # Example wiring to your path; adjust submodule=None for total causal module.
    GRAPHML = "/mnt/f/10_osteo_MR/results_network/largest_causal_subnet_A2_a6_g0.001832981/causal_modules.graphml"
    OUTDIR  = "/mnt/f/10_osteo_MR/gnn_from_graphml_runs"

    # Train on the total causal module with GAT
    trainer, preds = train_from_graphml(
        graphml_path=GRAPHML,
        outdir=OUTDIR,
        submodule=None,           # or an int 0..12
        model_type="GAT",
        hidden=128,
        layers=3,
        heads=4,
        dropout=0.3,
        attn_dropout=0.1,
        lr=1e-3,
        weight_decay=5e-4,
        epochs=300,
        patience=60,
        val_size=0.2,
        add_degree_features=True,
        use_label_as_feat=True,
        add_community_feature=False,
    )
    print("Saved:", os.path.join(OUTDIR, "GAT_suball_beta_predictions.tsv"))

