import gemmi
import numpy as np
import torch
from pathlib import Path
from Bio.Align import PairwiseAligner
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from multiprocessing import Pool
from parser import get_protein_info
import torch
from pathlib import Path
from tqdm import tqdm
from parser import get_protein_info
import gemmi
import numpy as np
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# =========================================================
# 🧬 Residue Mapping
# =========================================================

AA_MAP = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

def three_to_one_array(resnames):
    return np.array([AA_MAP.get(r, "X") for r in resnames])


# =========================================================
# 🧬 STRUCTURE PARSER (FINAL)
# =========================================================

def parse_structure(cif_path: Path, fasta_seq: str, chain_id: str):
    try:
        doc = gemmi.cif.read_file(str(cif_path))
        block = doc.sole_block()

        table = block.find([
            "_atom_site.label_atom_id",
            "_atom_site.label_seq_id",
            "_atom_site.label_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.occupancy"
        ])

        if table is None or len(table) == 0:
            logger.error(f"Table returned none for {cif_path.stem}")
            return None

        # =====================================================
        # Extract columns (vectorized)
        # =====================================================
        atom_name = np.array(table.column(0))
        raw_seq_id = np.array(table.column(1))

        # Convert safely: replace invalid entries with -1
        def safe_int(x):
            try:
                return int(x)
            except:
                return -1  # mark invalid

        label_seq_id = np.array([safe_int(x) for x in raw_seq_id], dtype=np.int32)
        res_name = np.array(table.column(2))
        chain = np.array(table.column(3))

        x = np.array(table.column(4), dtype=np.float32)
        y = np.array(table.column(5), dtype=np.float32)
        z = np.array(table.column(6), dtype=np.float32)
        coords_all = np.stack([x, y, z], axis=1)

        bfactor = np.array(table.column(7), dtype=np.float32)
        occupancy = np.array(table.column(8), dtype=np.float32)

        # =====================================================
        # Metadata
        # =====================================================
        methods = block.find_values("_exptl.method")
        method = "; ".join(methods) if methods else None
        is_alphafold = (method is None)

        resolution = block.find_value("_refine.ls_d_res_high")
        resolution = float(resolution) if resolution not in [None, "?", "."] else None

        # =====================================================
        # 🔥 Chain filtering (ONLY for PDB)
        # =====================================================
        if not is_alphafold:
            mask_chain = (chain == chain_id)
            if mask_chain.sum() == 0:
                return None

            atom_name = atom_name[mask_chain]
            label_seq_id = label_seq_id[mask_chain]
            res_name = res_name[mask_chain]
            coords_all = coords_all[mask_chain]
            bfactor = bfactor[mask_chain]
            occupancy = occupancy[mask_chain]

        # =====================================================
        # 🔥 Select CA atoms only
        # =====================================================
        ca_mask = (atom_name == "CA")

        if ca_mask.sum() == 0:
            logger.error(f"Ca_mask sum returned 0 for {cif_path.stem}")
            return None

        atom_name = atom_name[ca_mask]
        label_seq_id = label_seq_id[ca_mask]
        res_name = res_name[ca_mask]
        coords = coords_all[ca_mask]
        b_vals = bfactor[ca_mask]
        occ_vals = occupancy[ca_mask]

        # =====================================================
        # 🧬 Build CIF sequence (CA-based)
        # =====================================================
        cif_seq = "".join(three_to_one_array(res_name))
        L = len(fasta_seq)

        # =====================================================
        # 🔥 AlphaFold mapping via substring match
        # =====================================================
        if is_alphafold:

            matches = [
                i for i in range(len(cif_seq))
                if cif_seq.startswith(fasta_seq, i)
            ]

            if len(matches) != 1:
                logger.error(f"len(matches) returned none for {cif_path.stem}")
                return None

            start = matches[0]

            idx = (label_seq_id - 1) - start

        else:
            # PDB mapping
            idx = label_seq_id - 1

        # =====================================================
        # Bounds check
        # =====================================================
        valid = (idx >= 0) & (idx < L)

        if valid.sum() == 0:
            return None

        idx = idx[valid]
        coords = coords[valid]
        b_vals = b_vals[valid]
        occ_vals = occ_vals[valid]

        # =====================================================
        # 🧬 Build full-length arrays
        # =====================================================
        full_coords = np.zeros((L, 3), dtype=np.float32)
        full_b = np.zeros(L, dtype=np.float32)
        full_occ = np.zeros(L, dtype=np.float32)
        has_structure = np.zeros(L, dtype=bool)

        full_coords[idx] = coords
        full_b[idx] = b_vals
        full_occ[idx] = occ_vals
        has_structure[idx] = True

        has_confidence = has_structure.copy()

        # =====================================================
        # 🔥 Confidence computation
        # =====================================================
        conf = np.zeros(L, dtype=np.float32)

        if is_alphafold:
            # pLDDT normalization
            conf[has_structure] = full_b[has_structure] / 100.0
        else:
            valid_b = full_b[has_structure]

            if len(valid_b) == 0:
                return None

            median = np.median(valid_b)
            mad = np.median(np.abs(valid_b - median)) + 1e-6

            z = np.zeros_like(full_b)

            # Compute ONLY on valid positions
            z[has_structure] = (full_b[has_structure] - median) / mad

            # Optional: clip for safety
            z = np.clip(z, -50, 50)

            sigmoid = np.zeros_like(full_b)
            sigmoid[has_structure] = 1 / (1 + np.exp(z[has_structure]))

            local_conf = np.zeros_like(full_b)
            local_conf[has_structure] = full_occ[has_structure] * sigmoid[has_structure]

            global_conf = 1.0
            if resolution is not None:
                global_conf = 1.0 / (1.0 + max(0.0, resolution - 2.5))

            conf = global_conf * local_conf

        # =====================================================
        # Return
        # =====================================================
        return {
            "coords": torch.from_numpy(full_coords),
            "has_structure": torch.from_numpy(has_structure),
            "confidence": torch.from_numpy(conf.astype(np.float32)),
            "has_confidence": torch.from_numpy(has_confidence),
            "resolution": resolution,
            "is_alphafold": is_alphafold
        }

    except Exception as e:
        print(f"[ERROR] {cif_path}: {e}")
        return None

#Edge Builder
def build_radius_graph(coords, has_structure, cutoff=10):
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
    dist = torch.norm(diff, dim=1, keepdim=True).clamp(min=1e-8)
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

def process_single_entry(label, chain_id, cif_path, fasta_seq, cutoff=10):
    """
    Full pipeline for ONE protein:
    CIF → structure → graph

    Returns:
        (label, graph_dict) OR (label, None)
    """

    try:
        # ---- Extract chain ----
        # chain_id = extract_chain_id(label)

        # ---- Parse structure ----
        data = parse_structure(
            Path(cif_path),
            fasta_seq,
            chain_id=chain_id
        )

        if data is None:
            # logger.error(f"{label} failed. parse_structure returned None")
            return label, None

        # ---- Build graph ----
        edge_index, edge_attr = build_radius_graph(
            data["coords"],
            data["has_structure"],
            cutoff=cutoff
        )

        if edge_index is None:
            logger.error(f"{label} failed. edge_index is None")
            return label, None

        # ---- Construct graph ----
        graph = construct_graph(data, edge_index, edge_attr)

        return label, graph

    except Exception as e:
        logger.error(f"[ERROR] {label}: {e}")
        return label, None


def save_shard(shard_dict, shard_id, output_dir):
    """
    Save a single shard to disk.
    """
    path = Path(output_dir) / f"graph_shard_{shard_id:04d}.pt"
    torch.save(shard_dict, path)

def create_shard_build_dataset(dataset_type, split, fasta_path):
    dataset = []
    protein_info = get_protein_info(fasta_path)
    for protein in protein_info:
        protein_entry_id = protein["entry_id"]
        protein_full_id = protein["full_id"]
        sequence = protein["sequence"]
        chain_id = protein["chain"]
        cif_path = Path(f"structures/{dataset_type}/{dataset_type}_{split}/{protein_entry_id.lower()}.cif")
        dataset.append((protein_full_id, chain_id, cif_path, sequence))
    return dataset

def build_graph_shards(
    dataset,
    dataset_type,
    split,
    shard_size=1000,
    cutoff=10.0
):

    output_dir = f"graph_shards/{dataset_type}_graph_shards/{split}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = sorted(dataset, key=lambda x: x[0])

    current_shard = {}
    current_shard_id = 0
    label_to_shard = {}

    total_written = 0
    failed_parse = 0
    failed_graph = 0

    for label, chain_id, cif_path, fasta_seq in tqdm(dataset, desc="Building graph shards" ):

        label, graph = process_single_entry(
            label, chain_id, cif_path, fasta_seq, cutoff
        )

        if graph is None:
            # distinguish failure type (optional)
            failed_parse += 1
            continue

        current_shard[label] = graph
        label_to_shard[label] = current_shard_id
        total_written += 1

        if len(current_shard) >= shard_size:
            save_shard(current_shard, current_shard_id, output_dir)
            current_shard = {}
            current_shard_id += 1

    if len(current_shard) > 0:
        save_shard(current_shard, current_shard_id, output_dir)

    torch.save(label_to_shard, output_dir / "graph_index.pt")

    assert len(label_to_shard) == total_written

    print("\nFinished:")
    print(f"  Total graphs: {total_written}")
    print(f"  Failed: {failed_parse}")
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


# if __name__ == "__main__":
    # fasta_path = Path("xray_filtered_pdb/filtered_pdb_test.fasta")
    # protein_info = get_protein_info(fasta_path)
    # for protein in tqdm(protein_info[:2], desc = "experiment running"):
    #     protein_id = protein["entry_id"]
    #     sequence = protein["sequence"]
    #     chain_id = protein["chain"]
    #     path_to_cif_file = Path(f"structures/af/af_test/{protein_id.lower()}.cif")
    #     print(path_to_cif_file)
    #     structure_info = parse_structure(path_to_cif_file, sequence, chain_id)
    #     print(structure_info)
        # edge_idx, edge_attr = build_radius_graph(structure_info["coords"], structure_info["has_structure"])
        # coverage, mean_conf, max_conf = compute_global_stats(structure_info["has_structure"],structure_info["confidence"])



"""
Current flow:
cif → shard builder → disk
training → load shard → dict → convert (if needed) → model

{
    "representations": [...],   # (L,1280) fp16
    "coords": [...],            # (L,3) fp32
    "edge_index": [...],        # (2,E)
    "confidence": [...],        # (L,) fp32
    "has_confidence": [...],    # (L,) bool
    "has_structure": [...],     # (L,) bool
    "labels": [...],
    "method": [...],            # string or int
}

- Consider using resolution as a global_confidence_score. DONE ALREADY✅✅ (confirm sha. AI can be silly. Lmao)
This will change the definition of confidence_value:
global_conf = f(resolution)

final_conf_i = global_conf * local_conf_i

Example:

global_conf = 1 / (1 + (resolution - 2.5))
"""