import gemmi
import torch
import csv
import logging
import numpy as np
import parasail
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from reliability_aware.parser import get_protein_info
from torch_geometric.nn import radius_graph

logger = logging.getLogger(__name__)

# Residue mapping
AA_MAP = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

def three_to_one_array(resnames):
    return np.array([AA_MAP.get(r, "X") for r in resnames])

def build_aligned_graph_shards(
    manifest_path: str | Path,
    fasta_path: str | Path,
    structure_dir: str | Path,
    output_dir: str | Path,
    get_protein_info_fn,
    process_single_entry_fn,
    cutoff: float = 10.0,
):
    """
    Builds graph shards that align 1:1 with the extracted ESM shards and local indices.

      graph_shard_{k:04d}.pt["graphs"][i]
      corresponds to
      esm shard part_{k:04d}.pt["representations"][i]

    Notes:
    - Graph node arrays are truncated to manifest['truncated_length'] so they match
      truncated ESM residue embeddings.
    - Failed graph builds are stored as None to preserve alignment.
    """

    manifest_rows = _load_esm_manifest(manifest_path)
    output_dir = Path(output_dir).resolve()
    structure_dir = Path(structure_dir).resolve()

    label_lookup = _build_label_to_fasta_info(fasta_path, get_protein_info_fn)

    rows_by_shard = defaultdict(list)
    for row in manifest_rows:
        rows_by_shard[row["shard_number"]].append(row)
    total_rows = len(manifest_rows)
    total_ok = 0
    total_failed = 0
    with tqdm(total= total_rows, desc = "Building aligned graph shards: ", miniters=1, mininterval=0.0) as pbar: 
        for shard_id in sorted(rows_by_shard.keys()):
            rows = sorted(rows_by_shard[shard_id], key=lambda r: r["local_seq_idx"])

            # Enforce contiguous local indices
            expected = list(range(len(rows)))
            observed = [r["local_seq_idx"] for r in rows]
            if observed != expected:
                raise ValueError(
                    f"Shard {shard_id} local_seq_idx mismatch. "
                    f"Expected {expected[:5]}..., got {observed[:5]}..."
                )

            shard_graphs = []

            for row in rows:
                try: 
                    label = row["label"]
                    trunc_len = row["truncated_length"]

                    if label not in label_lookup:
                        logger.error(f"[MISSING FASTA INFO] {label}")
                        shard_graphs.append(None)
                        total_failed += 1
                        continue

                    info = label_lookup[label]
                    structure_file = structure_dir / f"{info['entry_id'].lower()}.cif"

                    _, graph = process_single_entry_fn(
                        label=label,
                        chain_id=info["chain_id"],
                        structure_file=structure_file,
                        fasta_seq=info["sequence"],
                        cutoff=cutoff,
                    )

                    if graph is None:
                        shard_graphs.append(None)
                        total_failed += 1
                        continue

                    shard_graphs.append(graph)
                    total_ok += 1
                finally:
                    pbar.update(1)
                

            _save_aligned_graph_shard(shard_graphs, shard_id, output_dir)

        metadata = {
            "num_manifest_rows": len(manifest_rows),
            "num_graphs_ok": total_ok,
            "num_graphs_failed": total_failed,
            "alignment": "graph_shard_k graphs[i] aligned to ESM shard k local_seq_idx i",
        }
        torch.save(metadata, output_dir / "graph_shard_metadata.pt")
        logger.info(f"Done! Graphs built: {total_ok} | Graphs failed: {total_failed} | Output dir: {output_dir}")

def process_single_entry(label, chain_id, structure_file, fasta_seq, cutoff=10):
    """
    Full pipeline for ONE protein:
    CIF → structure → graph

    Returns:
        (label, graph_dict) OR (label, None)
    """

    try:
        # Parse structure 
        data = parse_structure(
            Path(structure_file),
            fasta_seq,
            chain_id=chain_id
        )

        if data is None:
            return label, None

        # Build graph 
        edge_index, edge_attr = build_radius_graph(
            data["coords"],
            data["has_structure"],
            cutoff=cutoff
        )

        if edge_index is None:
            logger.error(f"{label} failed. edge_index is None")
            return label, None

        # Construct graph
        graph = construct_graph(data, edge_index, edge_attr)

        return label, graph

    except Exception as e:
        logger.error(f"[ERROR] {label}: {e}")
        return label, None


def parse_structure(structure_file, fasta_seq: str, chain_id: str):
    try:
        doc = gemmi.cif.read_file(str(structure_file))
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
            logger.error(f"parse_structure | [TABLE EMPTY] {structure_file.stem}")
            return None

        # Extract column information
        atom_name = np.array(table.column(0))
        raw_seq_id = np.array(table.column(1))

        def safe_int(x):
            try:
                return int(x)
            except:
                return -1

        label_seq_id = np.array([safe_int(x) for x in raw_seq_id], dtype=np.int32) 
        res_name = np.array(table.column(2))
        chain = np.array(table.column(3))

        x = np.array(table.column(4), dtype=np.float32) 
        y = np.array(table.column(5), dtype=np.float32)
        z = np.array(table.column(6), dtype=np.float32)
        coords_all = np.stack([x, y, z], axis=1)

        bfactor = np.array(table.column(7), dtype=np.float32)
        occupancy = np.array(table.column(8), dtype=np.float32)

        # Retrive metadata
        methods = block.find_values("_exptl.method")
        method = "; ".join(methods) if methods else None

        resolution = block.find_value("_refine.ls_d_res_high")
        resolution = float(resolution) if resolution not in [None, "?", "."] else None

    # Chain filtering for experimentally-derived structures (i.e. PDB only)
        mask_chain = (chain == chain_id)
        if mask_chain.sum() == 0:
            logger.error(f"parse_structure | [CHAIN FAIL] {structure_file.stem}: chain {chain_id}")
            return None

        atom_name = atom_name[mask_chain]
        label_seq_id = label_seq_id[mask_chain]
        res_name = res_name[mask_chain]
        coords_all = coords_all[mask_chain]
        bfactor = bfactor[mask_chain]
        occupancy = occupancy[mask_chain]

        # filter for alpha carbon (CA) atoms only
        ca_mask = (atom_name == "CA")

        if ca_mask.sum() == 0:
            logger.error(f"parse_structure | [NO CA] {structure_file.stem}")
            return None

        atom_name = atom_name[ca_mask]
        label_seq_id = label_seq_id[ca_mask]
        res_name = res_name[ca_mask]
        coords = coords_all[ca_mask]
        b_vals = bfactor[ca_mask]
        occ_vals = occupancy[ca_mask]

        # Build one-letter CIF sequence list from three-leter amino acid representation
        cif_seq = "".join(three_to_one_array(res_name))
        L = len(fasta_seq)
            
        idx = label_seq_id - 1 #zero-indexing

        # Bounds check (i.e. Return none if no sequence information is present in the fasta file)
        valid = (idx >= 0) & (idx < L)

        if valid.sum() == 0:
            logger.error(f"parse_structure | [INDEX FAIL] {structure_file.stem}")
            return None

        idx = idx[valid]
        coords = coords[valid]
        b_vals = b_vals[valid]
        occ_vals = occ_vals[valid]

        # Build full arrays for graph object
        full_coords = np.zeros((L, 3), dtype=np.float32)
        full_b_factor = np.zeros(L, dtype=np.float32)
        full_occupancy = np.zeros(L, dtype=np.float32)
        has_structure = np.zeros(L, dtype=bool)

        full_coords[idx] = coords
        full_b_factor[idx] = b_vals
        full_occupancy[idx] = occ_vals
        has_structure[idx] = True

        has_confidence = has_structure.copy()
        conf = _calculate_confidence(has_structure, full_b_factor, full_occupancy, L) #dim -> (L, )
        if conf is None:
            logger.error(f"parse_structure | [CONFIDENCE FAIL] {structure_file.stem}")
            return None
        
        return {
            "coords": torch.from_numpy(full_coords),
            "has_structure": torch.from_numpy(has_structure),
            "confidence": torch.from_numpy(conf.astype(np.float32)),
            "has_confidence": torch.from_numpy(has_confidence),
            "resolution": resolution
        }

    except Exception as e:
        logger.error(f"[EXCEPTION] {structure_file}: {e}")
        return None

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

def construct_graph(data, edge_index, edge_attr):
    """
    Construct final graph dictionary with all required metadata.
    """

    coords = data["coords"]
    has_structure = data["has_structure"]
    confidence = data["confidence"]
    resolution = data["resolution"]

    #Global stats
    coverage, mean_conf, std_conf = _compute_global_stats(
        has_structure, confidence
    )

    # Edge weights
    edge_weight = _compute_edge_weights(edge_index, confidence)


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
        "std_confidence": std_conf,               # float
        "resolution": resolution,

        # Experimental detail
        "resolution": data["resolution"],         # float or None
    }

    return graph

def _calculate_confidence(has_structure, full_b_factor, full_occupancy, sequence_length):
    L = sequence_length
    conf = np.zeros(L, dtype=np.float32)

    valid_b = full_b_factor[has_structure]

    if len(valid_b) == 0:
        return None

    median = np.median(valid_b)
    mad = np.median(np.abs(valid_b - median)) + 1e-6

    z = np.zeros_like(full_b_factor)
    z[has_structure] = (full_b_factor[has_structure] - median) / mad
    z = np.clip(z, -50, 50)

    sigmoid = np.zeros_like(full_b_factor)
    sigmoid[has_structure] = 1 / (1 + np.exp(z[has_structure]))

    local_conf = np.zeros_like(full_b_factor)
    local_conf[has_structure] = (
        full_occupancy[has_structure] * sigmoid[has_structure]
    )

    conf = local_conf

    return conf


def _compute_global_stats(has_structure, confidence):
    """
    Compute global structural statistics.
    """
    mask = has_structure.bool()

    if mask.sum() == 0:
        return 0.0, 0.0, 0.0

    conf_valid = confidence[mask]

    coverage = float(mask.float().mean())
    mean_conf = float(conf_valid.mean())
    std_conf = float(conf_valid.std())

    return coverage, mean_conf, std_conf


def _compute_edge_weights(edge_index, confidence):
    """
    Compute edge weights:
        w_ij = c_i * c_j
    """
    i, j = edge_index
    return confidence[i] * confidence[j]


def _load_esm_manifest(manifest_path: str | Path):
    """
    Reads the ESM manifest and returns rows in manifest order.

    Expected columns:
      - shard_number
      - local_seq_idx
      - global_seq_idx
      - label
      - sequence_length
      - was_truncated
      - truncated_length
    """
    manifest_path = Path(manifest_path).resolve()
    rows = []

    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "shard_number",
            "local_seq_idx",
            "global_seq_idx",
            "label",
            "sequence_length",
            "truncated_length",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        for row in reader:
            rows.append(
                {
                    "shard_number": int(row["shard_number"]),
                    "local_seq_idx": int(row["local_seq_idx"]),
                    "global_seq_idx": int(row["global_seq_idx"]),
                    "label": row["label"],
                    "sequence_length": int(row["sequence_length"]),
                    "truncated_length": int(row["truncated_length"]),
                }
            )

    return rows


def _build_label_to_fasta_info(fasta_path: str | Path, get_protein_info_fn):
    """
    Builds a lookup:
      label -> {sequence, chain_id, entry_id, full_id}
    """
    protein_info = get_protein_info_fn(Path(fasta_path).resolve())

    lookup = {}
    for protein in protein_info:
        label = protein["full_id"]   # must match manifest label
        lookup[label] = {
            "sequence": protein["sequence"],
            "chain_id": protein["chain"],
            "entry_id": protein["entry_id"],
            "full_id": protein["full_id"],
        }

    return lookup


def _save_aligned_graph_shard(
    shard_records: list,
    shard_id: int,
    output_dir: Path,
):
    """
    Saves one aligned graph shard.

    shard_records is a list where position i matches local_seq_idx=i.
    Each element is either:
      - graph dict
      - None if graph construction failed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"graph_shard_{shard_id:04d}.pt"
    torch.save(
        {
            "graphs": shard_records,
        },
        path,
    )

