import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

import logging


logger = logging.getLogger(__name__)


class ScalarAttentionPooling(nn.Module):
    """
    Scalar attention pooling over residue embeddings.

    Input:
        x: Tensor of shape (B, L, D)
        mask: Bool tensor of shape (B, L), where True = valid residue, False = padding
    Output:
        pooled: Tensor of shape (B, D)
        attn: Tensor of shape (B, L) with attention weights
    """

    def __init__(self, input_dim: int = 1280, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()

        if hidden_dim is None:
            self.scorer = nn.Linear(input_dim, 1)
        else:
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args: 
            x: (B, L, D)
            mask: (B, L) bool

        Returns:
            pooled: (B, D)
            attn: (B, L)
        """
        #Compute raw attention scores: (B, L, 1) -> (B, L)
        scores = self.scorer(x).squeeze(-1)

        #Mask out padding before softmax
        scores = scores.masked_fill(~mask, float(-1e9))

        #Attention weights over residues
        attn = torch.softmax(scores, dim = 1) # (B, L)

        #Weighted sum of residue embeddings
        pooled = torch.sum(x * attn.unsqueeze(-1), dim = 1) # (B, D)

        return pooled, attn

class ESMSequenceBranch(nn.Module):
    """
    Sequence branch:
    residue embeddings -> scalar attention pooling -> protein vector
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        attn_hidden_dim: int = 256,
        attn_dropout: float = 0.1,
        out_dim: int | None = None,
        ):
        super().__init__()
        
        self.pool = ScalarAttentionPooling(
            input_dim = esm_dim,
            hidden_dim = attn_hidden_dim,
            dropout = attn_dropout,
        )
        self.proj = None if out_dim is None else nn.Linear(esm_dim, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        pooled, attn = self.pool(x, mask) #pooled: (B, 1280)
        if self.proj is not None:
            pooled = self.proj(pooled)

        return pooled, attn


class ConfidenceWeightedAttentionPooling(nn.Module):
    """
    Pool node embeddings with:
        s_i = u^T tanh(W h_i + b)
        alpha_i = exp(s_i) * c_i / sum_j exp(s_j) * c_j
        r = sum_i alpha_i h_i

    Works on a PyG batched graph.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.score = nn.Linear(input_dim, input_dim)
        self.context = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, confidence: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            x:          (N, D) node embeddings after GAT
            confidence: (N,)   node confidence values c_i
            batch:      (N,)   graph assignment vector from PyG

        Returns:
            pooled:     (B, D) graph-level embeddings
            alpha:      (N,)   node pooling weights
        """
        # s_i = u^T tanh(W h_i + b)
        scores = self.context(torch.tanh(self.score(x))).squeeze(-1)  # (N,)

        # exp(s_i) * c_i
        weighted = torch.exp(scores) * confidence.clamp(min=1e-8)     # (N,)

        num_graphs = int(batch.max().item()) + 1
        denom = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
        denom.scatter_add_(0, batch, weighted)
        alpha = weighted / (denom[batch] + 1e-12)                     # (N,)

        pooled = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
        pooled.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), alpha.unsqueeze(-1) * x)

        return pooled, alpha


class GATBranch(nn.Module):
    """
    Graph branch:
      node features -> 2-layer GATv2 -> confidence-weighted attention pooling -> graph embedding

    Expected graph_batch fields:
      x             : (N, 1280) ESM residue embeddings
      confidence    : (N,)
      edge_index    : (2, E)
      edge_attr     : (E, 5)  [distance, dx, dy, dz, reliability_weight]
      batch         : (N,)
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 5,
        out_dim: int = 1280,
        use_confidence_as_node_feature: bool = True,
    ):
        super().__init__()

        self.use_confidence_as_node_feature = use_confidence_as_node_feature
        in_dim = esm_dim + 1 if use_confidence_as_node_feature else esm_dim

        self.gat1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=out_dim,
            heads=1,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        self.pool = ConfidenceWeightedAttentionPooling(out_dim)

    def forward(self, graph_batch):
        """
        Returns:
            graph_repr: (B, 1280)
            node_alpha: (N,) pooling weights
        """
        x = graph_batch.x.float()                       # (N, 1280)
        confidence = graph_batch.confidence.float()    # (N,)
        edge_index = graph_batch.edge_index.long()
        edge_attr = graph_batch.edge_attr.float()      # (E, 5)
        batch = graph_batch.batch.long()

        if self.use_confidence_as_node_feature:
            conf_feat = confidence.unsqueeze(-1)       # (N, 1)
            x = torch.cat([x, conf_feat], dim=-1)      # (N, 1281)

        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.norm1(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.norm2(x)
        x = self.dropout(x)

        graph_repr, node_alpha = self.pool(x, confidence, batch)
        return graph_repr, node_alpha

class FusionMLP(nn.Module):
    """
    Fuses sequence and graph representations:
        r^(n) = MLP([r^(s) | r^(g)])
    """

    def __init__(
        self,
        seq_dim: int = 1280,
        graph_dim: int = 1280,
        hidden_dim: int = 1024,
        out_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = seq_dim + graph_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, seq_repr: torch.Tensor, graph_repr: torch.Tensor):
        x = torch.cat([seq_repr, graph_repr], dim=-1)
        return self.net(x)

class NeuralLogitHead(nn.Module):
    """
    Maps fused neural representation r^(n) to neural logits z^(n).
    """

    def __init__(
        self,
        in_dim: int = 512,
        num_go_terms: int = 100, 
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_go_terms),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)



def collate_fn(batch):
    reps = [x[0] for x in batch]
    global_indices = torch.tensor([x[1] for x in batch], dtype=torch.long)

    lengths = [r.shape[0] for r in reps]
    max_len = max(lengths)
    dim = reps[0].shape[1]
    

    padded = torch.zeros(len(reps), max_len, dim, dtype=reps[0].dtype)
    mask = torch.zeros(len(reps), max_len, dtype=torch.bool)

    for i, r in enumerate(reps):
        L = r.shape[0]
        padded[i, :L] = r
        mask[i, :L] = True

    return padded, mask, global_indices





#Add to script file later - THIS VERSION WORKS IF I WANT TO DO A SEQUENCE ONLY RUN WITH THE SCALAR POOLED EMBEDDINGS (SO, TECHNICALLY THIS IS MY SEQUENCE ONLY BASELINE)
# dataset = ESMShardDataset(
#     shard_dir="path/to/shards",
#     manifest_path="path/to/manifest.csv",
#     cache_size=3,
# )

# BATCH_SIZE = 16
# batch_sampler = HybridBatchSampler(
#     dataset,
#     batch_size=BATCH_SIZE,
#     active_shards=3,
#     lookahead_factor=2,
#     drop_last=True,
#     seed=42,
# )

# loader = DataLoader(
#     dataset,
#     batch_sampler=batch_sampler,
#     collate_fn=collate_fn,
#     num_workers=0, #Performs best with 0. num_workers > 0 can cause unnecessary overhead since each worker gets its own shard cache
#     pin_memory=True,
# )

#SCRIPT FOR muLtimodal version
# from torch.utils.data import DataLoader

# dataset = ESMGraphMultimodalDataset(
#     esm_shard_dir="path/to/esm_shards",
#     graph_shard_dir="path/to/aligned_graph_shards",
#     manifest_path="path/to/manifest.csv",
#     esm_cache_size=2,
#     graph_cache_size=4,
#     require_graph=True,
# )

# batch_sampler = HybridBatchSampler(
#     dataset=dataset,
#     batch_size=16,
#     active_shards=3,
#     lookahead_factor=2,
#     drop_last=True,
#     seed=42,
# )

# loader = DataLoader(
#     dataset,
#     batch_sampler=batch_sampler,
#     collate_fn=multimodal_collate_fn,
#     num_workers=0,
#     pin_memory=True,
# )

# seq_branch = ESMSequenceBranch()

# gat_branch = GATBranch(
#     esm_dim=1280,
#     hidden_dim=256,
#     heads=4,
#     dropout=0.1,
#     edge_dim=5,
#     out_dim=1280,
#     use_confidence_as_node_feature=True,
# )

# num_go_terms = 500  # replace with your real output dimension

# model = ReliabilityAwareProteinFunctionModel(
#     seq_branch=seq_branch,
#     gat_branch=gat_branch,
#     num_classes=num_go_terms,
#     fusion_hidden_dim=1024,
#     fusion_out_dim=512,
#     gate_q_dim=9,
#     gate_hidden_dim=32,
#     dropout=0.2,
# ).to(device)


#EXAMPLE TRAINING ONE EPOCH
# for padded, mask, graph_batch, global_indices, labels in loader:
#     outputs = model(
#     padded=padded.to(device),
#     mask=mask.to(device),
#     graph_batch=graph_batch.to(device),
#     homology_logits=homology_logits.to(device),   # (B, C)
#     gate_features=gate_features.to(device),       # (B, 9)
# )

# final_logits = outputs["final_logits"]
