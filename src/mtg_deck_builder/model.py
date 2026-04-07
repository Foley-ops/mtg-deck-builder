"""GATv2 model and training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from mtg_deck_builder.card import Card
from mtg_deck_builder.config import DEVICE, EMBED_DIM


class CardGNN(nn.Module):
    """GATv2 -> L2-normalized card embeddings."""

    def __init__(self, in_dim, hidden_dim, embed_dim, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            in_ch = hidden_dim if i == 0 else hidden_dim * heads
            self.convs.append(
                GATv2Conv(
                    in_ch, hidden_dim, heads=heads, dropout=dropout, add_self_loops=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim * heads))
        self.output_proj = nn.Linear(hidden_dim * heads, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.gelu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h_new = F.gelu(norm(conv(h, edge_index)))
            h = (
                (h + self.dropout(h_new))
                if h.shape == h_new.shape
                else self.dropout(h_new)
            )
        return F.normalize(self.output_proj(h), p=2, dim=-1)


class SynergyPredictor(nn.Module):
    """GNN + archetype-conditioned deck scoring."""

    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
        self.archetype_embeddings = nn.Embedding(8, EMBED_DIM)
        self.archetype_gate = nn.Linear(EMBED_DIM * 2, EMBED_DIM)

    def get_embeddings(self, x, edge_index, edge_attr=None):
        return self.gnn(x, edge_index, edge_attr)

    def predict_edge(self, embeddings, src, dst):
        return (embeddings[src] * embeddings[dst]).sum(dim=-1)

    def score_deck_candidates(self, embeddings, commander_idx, archetype_idx=0):
        cmd = embeddings[commander_idx].unsqueeze(0)
        arch = self.archetype_embeddings(
            torch.tensor(archetype_idx, device=embeddings.device)
        ).unsqueeze(0)
        combined = torch.cat(
            [cmd.expand_as(embeddings), arch.expand(embeddings.size(0), -1)], dim=-1
        )
        gate = torch.sigmoid(self.archetype_gate(combined))
        query = gate * cmd + (1 - gate) * arch
        return (embeddings * query).sum(dim=-1)


def train_gnn(model, data, edge_weights, epochs=200, lr=1e-3, neg_ratio=5):
    from tqdm import tqdm

    model.to(DEVICE)
    data = data.to(DEVICE)
    edge_weights = edge_weights.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    num_nodes = data.x.size(0)
    losses = []
    model.train()

    pbar = tqdm(range(epochs), desc="  Training", unit="ep",
                bar_format="  {l_bar}{bar:30}{r_bar}")

    for epoch in pbar:
        optimizer.zero_grad()
        emb = model.get_embeddings(data.x, data.edge_index)
        src, dst = data.edge_index[0], data.edge_index[1]
        w = torch.log1p(edge_weights).clamp(min=0.1)
        pos_loss = -torch.mean(w * F.logsigmoid(model.predict_edge(emb, src, dst)))
        neg_src = src.repeat(neg_ratio)
        neg_dst = torch.randint(0, num_nodes, (len(neg_src),), device=DEVICE)
        neg_loss = -torch.mean(F.logsigmoid(-model.predict_edge(emb, neg_src, neg_dst)))
        ae = model.archetype_embeddings.weight
        arch_reg = (
            (torch.mm(ae, ae.t()) * (1 - torch.eye(ae.size(0), device=DEVICE))) ** 2
        ).mean() * 0.1
        loss = pos_loss + neg_loss + arch_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

    return losses
