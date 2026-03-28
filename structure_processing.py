import gemmi
import numpy as np
import torch
from pathlib import Path
from Bio.Align import PairwiseAligner
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from multiprocessing import Pool

import torch
from pathlib import Path
from tqdm import tqdm

#Global Aligner
ALIGNER = PairwiseAligner()
ALIGNER.mode = "global"
ALIGNER.match_score = 2
ALIGNER.mismatch_score = -1
ALIGNER.open_gap_score = -5
ALIGNER.extend_gap_score = -0.5


#Residue Mapping
AA_MAP = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}


def three_to_one(resnames):
    """Convert 3-letter residue names to 1-letter sequence."""
    return "".join([AA_MAP.get(r, "X") for r in resnames])


#Alignment
def fast_subsequence_match(long_seq, short_seq):
    """Fast exact subsequence match."""
    start = long_seq.find(short_seq)
    if start == -1:
        return None
    return {i: start + i for i in range(len(short_seq))}


def pairwise_aligner_map(long_seq, short_seq):
    """Fallback alignment using Biopython PairwiseAligner."""
    alignments = ALIGNER.align(long_seq, short_seq)
    if len(alignments) == 0:
        return None

    aln = alignments[0]
    long_blocks, short_blocks = aln.aligned

    mapping = {}
    for (l_start, l_end), (s_start, s_end) in zip(long_blocks, short_blocks):
        for i in range(s_end - s_start):
            mapping[s_start + i] = l_start + i

    return mapping if mapping else None


def align_sequences(long_seq, short_seq):
    """Hybrid alignment: fast exact match → global alignment fallback."""
    match = fast_subsequence_match(long_seq, short_seq)
    if match is not None:
        return match
    return pairwise_aligner_map(long_seq, short_seq)


#Structure parser
def parse_structure(cif_path: Path, fasta_seq: str, chain_id: str = None, eps: float = 1e-6):
    """
    Parse mmCIF file and align structure to full FASTA sequence.

    Confidence handling:
    - AlphaFold: pLDDT / 100
    - PDB: occupancy * sigmoid(-zscore(B-factor)) using MAD normalization
      + optional global resolution scaling
    """
    try:
        doc = gemmi.cif.read_file(str(cif_path))
        block = doc.sole_block()

        atom_site = block.find_loop("_atom_site.")

        atom_name = np.array(atom_site.get_column("_atom_site.label_atom_id"))
        res_id = np.array(atom_site.get_column("_atom_site.label_seq_id"), dtype=np.int32)
        res_name = np.array(atom_site.get_column("_atom_site.label_comp_id"))
        chain = np.array(atom_site.get_column("_atom_site.auth_asym_id"))

        x = np.array(atom_site.get_column("_atom_site.Cartn_x"), dtype=np.float32)
        y = np.array(atom_site.get_column("_atom_site.Cartn_y"), dtype=np.float32)
        z = np.array(atom_site.get_column("_atom_site.Cartn_z"), dtype=np.float32)

        coords = np.stack([x, y, z], axis=1)

        bfactor = np.array(atom_site.get_column("_atom_site.B_iso_or_equiv"), dtype=np.float32)
        occupancy = np.array(atom_site.get_column("_atom_site.occupancy"), dtype=np.float32)

        # detect method
        methods = block.find_values("_exptl.method")
        method = "; ".join(methods) if methods else None
        is_alphafold = (method is None)

        # resolution (PDB only)
        resolution = block.find_value("_refine.ls_d_res_high")
        resolution = float(resolution) if resolution not in [None, "?", "."] else None

        mask_chain = (chain == chain_id) if chain_id else np.ones(len(chain), dtype=bool)

        ca_mask = (atom_name == "CA") & mask_chain
        if ca_mask.sum() == 0:
            return None

        res_ids = res_id[ca_mask]
        res_names = res_name[ca_mask]
        coords = coords[ca_mask]
        bfactor = bfactor[ca_mask]
        occupancy = occupancy[ca_mask]

        order = np.argsort(res_ids)
        res_ids = res_ids[order]
        res_names = res_names[order]
        coords = coords[order]
        bfactor = bfactor[order]
        occupancy = occupancy[order]

        cif_seq = three_to_one(res_names)
        mapping = align_sequences(cif_seq, fasta_seq)
        if mapping is None:
            return None

        L = len(fasta_seq)

        full_coords = np.zeros((L, 3), dtype=np.float32)
        full_b = np.zeros(L, dtype=np.float32)
        full_occ = np.zeros(L, dtype=np.float32)
        has_structure = np.zeros(L, dtype=bool)

        for fasta_idx, cif_idx in mapping.items():
            if cif_idx < len(coords):
                full_coords[fasta_idx] = coords[cif_idx]
                full_b[fasta_idx] = bfactor[cif_idx]
                full_occ[fasta_idx] = occupancy[cif_idx]
                has_structure[fasta_idx] = True

        # confidence
        conf = np.zeros(L, dtype=np.float32)
        has_confidence = has_structure.copy()

        if is_alphafold:
            conf[has_structure] = full_b[has_structure] / 100.0
        else:
            valid_b = full_b[has_structure]
            if len(valid_b) == 0:
                return None

            median = np.median(valid_b)
            mad = np.median(np.abs(valid_b - median)) + eps

            z = (full_b - median) / mad
            sigmoid = 1 / (1 + np.exp(z))  # equals σ(-z)

            local_conf = full_occ * sigmoid

            # global resolution scaling
            global_conf = 1.0
            if resolution is not None:
                global_conf = 1.0 / (1.0 + max(0.0, resolution - 2.5))

            conf = global_conf * local_conf

        return {
            "coords": torch.from_numpy(full_coords),
            "has_structure": torch.from_numpy(has_structure),
            "confidence": torch.from_numpy(conf.astype(np.float32)),
            "has_confidence": torch.from_numpy(has_confidence),
            "resolution": resolution,
            "is_alphafold": is_alphafold
        }

    except Exception:
        return None


#Edge Builder
def build_radius_graph(coords, has_structure, cutoff=10.0):
    """
    Build undirected radius graph using PyTorch Geometric.

    Returns:
    - edge_index (2, E)
    - edge_attr (E, 4): [distance, dx, dy, dz]
    """

    mask = has_structure.bool()
    coords_valid = coords[mask]

    if coords_valid.shape[0] == 0:
        return None, None

    edge_index = radius_graph(coords_valid, r=cutoff, loop=False)

    row, col = edge_index
    diff = coords_valid[col] - coords_valid[row]
    dist = torch.norm(diff, dim=1, keepdim=True)
    direction = diff / dist

    edge_attr = torch.cat([dist, direction], dim=1)

    valid_idx = torch.where(mask)[0]
    edge_index = valid_idx[edge_index]

    return edge_index, edge_attr


def extract_chain_id(label: str):
    """
    Extract chain ID from label.
    Example: 15Y3-A → A
    """
    if "-" in label:
        return label.split("-")[-1]
    return None


def compute_global_stats(has_structure, confidence):
    """
    Compute global structural statistics.
    """
    mask = has_structure.bool()

    if mask.sum() == 0:
        return 0.0, 0.0, 0.0

    conf_valid = confidence[mask]

    coverage = float(mask.float().mean())
    mean_conf = float(conf_valid.mean())
    max_conf = float(conf_valid.max())

    return coverage, mean_conf, max_conf


def compute_edge_weights(edge_index, confidence):
    """
    Compute edge weights:
        w_ij = c_i * c_j
    """
    row, col = edge_index
    return confidence[row] * confidence[col]


def construct_graph(data, edge_index, edge_attr):
    """
    Construct final graph dictionary with all required metadata.
    """

    coords = data["coords"]
    has_structure = data["has_structure"]
    confidence = data["confidence"]

    # ---- Global stats ----
    coverage, mean_conf, max_conf = compute_global_stats(
        has_structure, confidence
    )

    # ---- Edge weights ----
    edge_weight = compute_edge_weights(edge_index, confidence)

    graph = {
        # Core graph
        "coords": coords,                         # (L, 3)
        "edge_index": edge_index,                 # (2, E)
        "edge_attr": edge_attr,                   # (E, 4)
        "edge_weight": edge_weight,               # (E,)

        # Node-level metadata
        "has_structure": has_structure,           # (L,)
        "confidence": confidence,                 # (L,)

        # Global metadata
        "coverage": coverage,                     # float
        "mean_confidence": mean_conf,             # float
        "max_confidence": max_conf,               # float

        # Source info
        "is_alphafold": data["is_alphafold"],     # bool
        "is_experimental": not data["is_alphafold"],

        # Experimental detail
        "resolution": data["resolution"],         # float or None
    }

    return graph


def save_shard(shard_dict, shard_id, output_dir):
    """
    Save a single shard to disk.
    """
    path = Path(output_dir) / f"graph_shard_{shard_id:04d}.pt"
    torch.save(shard_dict, path)


def build_graph_shards(
    dataset,                  # List[Tuple[label, cif_path, fasta_seq]]
    output_dir,
    shard_size=1000,
    cutoff=10.0
):
    """
    Build structural graph shards.

    Each shard:
        Dict[label → graph_dict]

    Also produces:
        graph_index.pt → Dict[label → shard_id]

    Design guarantees:
    - deterministic shard assignment (sorted labels)
    - no failed entries included
    - memory-efficient (one shard at a time)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Deterministic ordering ----
    dataset = sorted(dataset, key=lambda x: x[0])

    current_shard = {}
    current_shard_id = 0
    label_to_shard = {}

    total_written = 0
    failed_parse = 0
    failed_graph = 0

    for label, cif_path, fasta_seq in tqdm(dataset):

        chain_id = extract_chain_id(label)

        # ---- Parse structure ----
        data = parse_structure(
            Path(cif_path),
            fasta_seq,
            chain_id=chain_id
        )

        if data is None:
            failed_parse += 1
            continue

        # ---- Build graph ----
        edge_index, edge_attr = build_radius_graph(
            data["coords"],
            data["has_structure"],
            cutoff=cutoff
        )

        if edge_index is None:
            failed_graph += 1
            continue

        # ---- Construct graph ----
        graph = construct_graph(data, edge_index, edge_attr)

        # ---- Add to shard ----
        current_shard[label] = graph
        label_to_shard[label] = current_shard_id
        total_written += 1

        # ---- Save shard if full ----
        if len(current_shard) >= shard_size:
            save_shard(current_shard, current_shard_id, output_dir)
            current_shard = {}
            current_shard_id += 1

    # ---- Save final shard ----
    if len(current_shard) > 0:
        save_shard(current_shard, current_shard_id, output_dir)

    # ---- Save index ----
    index_path = output_dir / "graph_index.pt"
    torch.save(label_to_shard, index_path)

    # ---- Sanity check ----
    assert len(label_to_shard) == total_written, \
        "Mismatch between index and written graphs"

    print("\nFinished:")
    print(f"  Total graphs: {total_written}")
    print(f"  Failed (parse): {failed_parse}")
    print(f"  Failed (graph): {failed_graph}")
    print(f"  Total shards: {current_shard_id + 1}")


def validate_random_samples(output_dir, num_samples=10):
    """
    Validate that shard lookup works correctly.
    """

    import random

    output_dir = Path(output_dir)
    index = torch.load(output_dir / "graph_index.pt")

    labels = list(index.keys())
    samples = random.sample(labels, min(num_samples, len(labels)))

    for label in samples:
        shard_id = index[label]
        shard_path = output_dir / f"graph_shard_{shard_id:04d}.pt"

        shard = torch.load(shard_path)

        assert label in shard, f"{label} missing in shard {shard_id}"

    print(f"Validation passed for {len(samples)} samples.")






