import gemmi
import numpy as np
import torch
from pathlib import Path
from Bio.Align import PairwiseAligner
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from multiprocessing import Pool

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


#Pipeline entry
def process_structure(cif_path, fasta_seq, cutoff=10.0):
    """End-to-end processing into PyG Data object."""

    data = parse_structure(cif_path, fasta_seq)
    if data is None:
        return None

    edge_index, edge_attr = build_radius_graph(
        data["coords"], data["has_structure"], cutoff
    )

    if edge_index is None:
        return None

    pyg_data = Data(
        x=data["coords"],
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    pyg_data.has_structure = data["has_structure"]
    pyg_data.confidence = data["confidence"]
    pyg_data.has_confidence = data["has_confidence"]
    pyg_data.resolution = data["resolution"]
    pyg_data.is_alphafold = data["is_alphafold"]

    return pyg_data


#Batch Processing
def process_many(inputs, num_workers=8, cutoff=10.0):
    """Parallel processing of multiple structures."""

    args = [(cif, fasta, cutoff) for cif, fasta in inputs]

    with Pool(num_workers) as p:
        results = p.starmap(process_structure, args)

    return [r for r in results if r is not None]


#I need a function to extract the coords

# def build_edges(coords, cutoff=10.0):
#     # coords: (L, 3)
#     dists = torch.cdist(coords, coords)  # (L, L)
#     edge_index = (dists < cutoff).nonzero(as_tuple=False).t()
#     return edge_index


"""
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

- Consider using resolution as a global_confidence_score. 
This will change the definition of confidence_value:
global_conf = f(resolution)

final_conf_i = global_conf * local_conf_i

Example:

global_conf = 1 / (1 + (resolution - 2.5))
"""