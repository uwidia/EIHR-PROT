import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from collections import deque
import random
import logging
from torch_geometric.data import Data, Batch
from reliability_aware.go_term_extraction import build_go_annotations_list
from reliability_aware.pool_embeddings import FusionMLP, NeuralLogitHead

logger = logging.getLogger(__name__)
GO_ASPECT = "BP"

class ReliabilityAwareProteinFunctionModel(nn.Module):
    """
    Full model:
      1) sequence branch -> r^(s)
      2) graph branch    -> r^(g)
      3) fusion MLP      -> r^(n)
      4) neural head     -> z^(n)
      5) neural probabilities -> sigmoid(z^(n)) -> p^(n)
      6) homology scores per go term -> p^(h)
      7) reliability gate on q -> [alpha_n, alpha_h]
      8) fused probabilities    -> z = alpha_n * neural_probs + alpha_h * p^(h)

    Expects:
      - seq_branch forward(padded, mask) -> (B, 1280), attn
      - gat_branch forward(graph_batch)  -> (B, 1280), node_alpha
    """

    def __init__(
        self,
        seq_branch: nn.Module,
        gat_branch: nn.Module,
        num_go_terms: int,
        fusion_hidden_dim: int = 1024,
        fusion_out_dim: int = 512,
        gate_q_dim: int = 8,
        gate_hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.seq_branch = seq_branch
        self.gat_branch = gat_branch

        self.fusion = FusionMLP(
            seq_dim=1280,
            graph_dim=1280,
            hidden_dim=fusion_hidden_dim,
            out_dim=fusion_out_dim,
            dropout=dropout,
        )

        self.neural_head = NeuralLogitHead(
            in_dim=fusion_out_dim,
            num_go_terms=num_go_terms,
            dropout=dropout,
        )

        self.gate = ReliabilityGate(
            q_dim=gate_q_dim,
            hidden_dim=gate_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        padded: torch.Tensor,
        mask: torch.Tensor,
        graph_batch,
        homology_scores: torch.Tensor,
        gate_features: torch.Tensor,
    ):
        """
        Args:
            padded:          (B, Lmax, 1280)
            mask:            (B, Lmax)
            graph_batch:     PyG batch
            homology_scores: (B, C)  -> z^(h)
            gate_features:   (B, 8)  -> q

        Returns:
            dict with:
              fused_probs      : (B, C)
              neural_logits     : (B, C)
              homology_scores   : (B, C)
              gate_weights      : (B, 2)
              seq_repr          : (B, 1280)
              graph_repr        : (B, 1280)
              fused_repr        : (B, 512)
              seq_attn          : (B, Lmax)
              graph_node_alpha  : (N,)
        """
        seq_repr, seq_attn = self.seq_branch(padded, mask)          # (B, 1280)
        graph_repr, graph_node_alpha = self.gat_branch(graph_batch) # (B, 1280)

        fused_repr = self.fusion(seq_repr, graph_repr)              # (B, 512)
        neural_logits = self.neural_head(fused_repr)                # (B, C)
        neural_probs = torch.sigmoid(neural_logits)                 # (B, C)

        gate_weights = self.gate(gate_features)                     # (B, 2)
        alpha_n = gate_weights[:, 0].unsqueeze(-1)                  # (B, 1)
        alpha_h = gate_weights[:, 1].unsqueeze(-1)                  # (B, 1)

        fused_probs = alpha_n * neural_probs + alpha_h * homology_scores

        
        return {
            "fused_probs": fused_probs,
            "neural_logits": neural_logits,
            "homology_scores": homology_scores,
            "gate_weights": gate_weights,
            "seq_repr": seq_repr,
            "graph_repr": graph_repr,
            "fused_repr": fused_repr,
            "seq_attn": seq_attn,
            "graph_node_alpha": graph_node_alpha,
        }

class ReliabilityGate(nn.Module):
    """
    Computes:
        [alpha_n, alpha_h] = softmax(MLP(q))

    where q is the vector of external reliability features.
    """

    def __init__(
        self,
        q_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(q_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, q: torch.Tensor):
        """
        Args:
            q: (B, q_dim)

        Returns:
            gate_weights: (B, 2) where [:,0]=alpha_n, [:,1]=alpha_h
        """
        logits = self.net(q)
        gate_weights = torch.softmax(logits, dim=-1)
        return gate_weights

class HybridBatchSampler(Sampler):
    """
    Custom Batch Sampler. Minimizes shard file I/O bottleneck during dataloading
    - keeps a small active pool of shards
    - shuffles within each shard every epoch
    - builds each batch from the active pool only
    - sorts local candidates by length and take a random window
    - brings in new shards as active ones empty out
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        active_shards: int = 3,
        lookahead_factor: int = 2,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.active_shards = min(active_shards, len(dataset.indices_by_shard))
        self.lookahead = batch_size * lookahead_factor
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        shard_order = list(self.dataset.indices_by_shard.keys())
        rng.shuffle(shard_order)

        shard_queues = {}
        for shard_id in shard_order:
            idxs = list(self.dataset.indices_by_shard[shard_id])
            rng.shuffle(idxs)
            shard_queues[shard_id] = deque(idxs)

        remaining_shards = deque(shard_order)
        active = []

        def refill():
            while len(active) < self.active_shards and remaining_shards:
                active.append(remaining_shards.popleft())

        refill()

        while active:
            active[:] = [sid for sid in active if shard_queues[sid]]
            refill()
            if not active:
                break

            candidates = []
            for sid in active:
                candidates.extend(list(shard_queues[sid])[:self.lookahead])

            if not candidates:
                break

            candidates.sort(key=lambda idx: self.dataset.lengths[idx])

            if len(candidates) <= self.batch_size:
                batch = candidates
            else:
                start = rng.randint(0, len(candidates) - self.batch_size)
                batch = candidates[start:start + self.batch_size]

            if len(batch) < self.batch_size and self.drop_last:
                break

            chosen = set(batch)
            for sid in active:
                shard_queues[sid] = deque(idx for idx in shard_queues[sid] if idx not in chosen)

            yield batch

def build_multihot_target_lookup(label_to_go_terms, go_terms):
    go_to_idx = {go: i for i, go in enumerate(go_terms)}
    num_terms = len(go_terms)

    label_to_target = {}

    for label, terms in label_to_go_terms.items():
        y = torch.zeros(num_terms, dtype=torch.float32)

        idxs = [go_to_idx[t] for t in terms if t in go_to_idx]
        if idxs:
            y[idxs] = 1.0

        label_to_target[label] = y

    return label_to_target

def multimodal_collate_fn_generator(label_to_indices, num_go_terms):
    def multimodal_collate_fn(batch):
        reps = [item["rep"] for item in batch]
        graphs = [item["graph"] for item in batch]
        labels = [item["label"] for item in batch]
        global_indices = torch.tensor([item["global_idx"] for item in batch], dtype=torch.long)

        lengths = [r.shape[0] for r in reps]
        max_len = max(lengths)
        dim = reps[0].shape[1]
        dtype = reps[0].dtype

        padded = torch.zeros(len(reps), max_len, dim, dtype=dtype)
        mask = torch.zeros(len(reps), max_len, dtype=torch.bool)

        for i, r in enumerate(reps):
            L = r.shape[0]
            padded[i, :L] = r
            mask[i, :L] = True

        pyg_graphs = []
        homology_priors = []
        gate_features = []

        targets = torch.zeros(len(batch), num_go_terms, dtype=torch.float32)

        for i, (item, rep, graph, label) in enumerate(zip(batch, reps, graphs, labels)):
            if graph is None:
                raise ValueError(f"Graph is None for label={label}")

            if rep.shape[0] != graph["coords"].shape[0]:
                raise ValueError(
                    f"Length mismatch for {label}: rep has {rep.shape[0]} residues, "
                    f"graph has {graph['coords'].shape[0]}"
                )

            edge_attr = torch.cat(
                [
                    graph["edge_attr"].float(),
                    graph["edge_weight"].float().unsqueeze(1),
                ],
                dim=1,
            )

            data = Data(
                x=rep.float(),
                confidence=graph["confidence"].float(),
                edge_index=graph["edge_index"].long(),
                edge_attr=edge_attr,
                label=label,
            )
            pyg_graphs.append(data)

            homology_priors.append(item["homology_prior"].float())

            mean_conf = torch.tensor(float(graph["mean_confidence"]), dtype=torch.float32)
            std_conf = torch.tensor(float(graph["std_confidence"]), dtype=torch.float32)
            coverage = torch.tensor(float(graph["coverage"]), dtype=torch.float32)

            resolution = graph["resolution"]
            r_pdb = 0.0 if resolution is None else float(resolution)
            r_pdb = torch.tensor([r_pdb], dtype=torch.float32)

            graph_q = torch.stack([mean_conf, std_conf, coverage])
            homology_q = item["homology_gate"].float()

            q = torch.cat([graph_q, homology_q, r_pdb], dim=0)
            gate_features.append(q)

            idxs = label_to_indices.get(label)
            if idxs is None:
                raise KeyError(f"No GO target found for label={label}")
            if idxs:
                targets[i, idxs] = 1.0

        graph_batch = Batch.from_data_list(pyg_graphs)
        homology_priors = torch.stack(homology_priors, dim=0)
        gate_features = torch.stack(gate_features, dim=0)

        return padded, mask, graph_batch, homology_priors, gate_features, targets, global_indices, labels

    return multimodal_collate_fn