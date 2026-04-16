import gemmi
import torch
import csv
import logging
import numpy as np
import parasail
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from reliability_aware.structure_processing import process_single_entry, compute_global_stats
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

    total_ok = 0
    total_failed = 0

    for shard_id in tqdm(sorted(rows_by_shard.keys()), desc="Building aligned graph shards"):
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
            label = row["label"]
            trunc_len = row["truncated_length"]

            if label not in label_lookup:
                logger.error(f"[MISSING FASTA INFO] {label}")
                shard_graphs.append(None)
                total_failed += 1
                continue

            info = label_lookup[label]
            structure_file = structure_dir / f"{info['entry_id'].lower()}.cif"

            label_out, graph = process_single_entry_fn(
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

            # truncate graph tensors to match truncated ESM embeddings
            graph = _truncate_graph_to_length(graph, trunc_len)

            shard_graphs.append(graph)
            total_ok += 1

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
        is_alphafold = (method is None)

        resolution = block.find_value("_refine.ls_d_res_high")
        resolution = float(resolution) if resolution not in [None, "?", "."] else None

        # Chain filtering for experimentally-derived structures (i.e. PDB only)
        if not is_alphafold:
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

        # Mapping of indices (for alphafold sequences with misalignment between cif_file sequence and curated protein sequences)
        if is_alphafold:
            idx = _perform_mapping(cif_seq, fasta_seq, structure_file.stem)
            
        else:
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
        conf = _calculate_confidence(is_alphafold, has_structure, full_b_factor, full_occupancy) #dim -> (L, )
        
        return {
            "coords": torch.from_numpy(full_coords),
            "has_structure": torch.from_numpy(has_structure),
            "confidence": torch.from_numpy(conf.astype(np.float32)),
            "has_confidence": torch.from_numpy(has_confidence),
            "resolution": resolution,
            "is_alphafold": is_alphafold
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

    # ---- Global stats ----
    coverage, mean_conf, std_conf, max_conf = _compute_global_stats(
        has_structure, confidence
    )

    # ---- Edge weights ----
    edge_weight = _compute_edge_weights(edge_index, confidence)

    #‼️‼️‼️‼️consider concating edge weight to attributes here then removing edge weight from the graph dict below
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
        "std_confidence": std_conf,
        "max_confidence": max_conf,               # float

        # Source info
        "is_alphafold": data["is_alphafold"],     # bool
        "is_experimental": not data["is_alphafold"],

        # Experimental detail
        "resolution": data["resolution"],         # float or None
    }

    return graph

#Map fasta and cif sequence indices
def _perform_mapping(cif_seq:str, fasta_seq:str, protein_id: str):
    matches = [
        i for i in range(len(cif_seq))
        if cif_seq.startswith(fasta_seq, i)
    ]

    if len(matches) == 1:
        start = matches[0]
        idx = (label_seq_id - 1) - start

    else:
        logger.warning(
            f"[FALLBACK ALIGN] {protein_id}: matches={len(matches)}"
        )

        result = _map_fasta_to_cif_parasail(
            fasta_seq,
            cif_seq,
            protein_id=protein_id
        )

        if result is None:
            logger.error(f"[FINAL FAIL] {protein_id}")
            return None

        _, inv_mapping = result

        idx = np.array([
            inv_mapping.get(seq_id - 1, -1)
            for seq_id in label_seq_id
        ], dtype=np.int32)
    return idx

# Sequence alignment between fasta sequence and cif file sequence
def _map_fasta_to_cif_parasail(fasta_seq:str, cif_seq:str, protein_id:str = None):
    matrix = parasail.blosum62

    result = parasail.nw_trace_striped_16(
        fasta_seq,
        cif_seq,
        5,   # gap open
        1,   # gap extend
        matrix
    )

    if result.traceback is None:
        logger.error(f"[ALIGN FAIL] {protein_id}: no traceback")

    aligned_fasta = result.traceback.query
    aligned_cif = result.traceback.ref

    logger.warning(f"[ALIGN USED] {protein_id}: score={result.score}")
    mapping = {}
    inv_mapping = {}

    f_idx = 0
    c_idx = 0

    for a, b in zip(aligned_fasta, aligned_cif):
        if a != "-" and b != "-":
            mapping[f_idx] = c_idx
            inv_mapping[c_idx] = f_idx

        if a != "-":
            f_idx += 1
        if b != "-":
            c_idx += 1

    coverage = len(mapping) / len(fasta_seq)

    if coverage < 0.7:
        logger.error(f"[ALIGN REJECTED] {protein_id}: coverage={coverage:.2f}")
        return None

    return mapping, inv_mapping

def _calculate_confidence(is_alphafold: bool, has_structure, full_b_factor, full_occupancy):
    conf = np.zeros(L, dtype=np.float32)

    if is_alphafold:
        conf[has_structure] = full_b_factor[has_structure] / 100.0
    else:
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

        global_conf = 1.0
        if resolution is not None:
            global_conf = 1.0 / (1.0 + max(0.0, resolution - 2.5))

        conf = global_conf * local_conf
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
    max_conf = float(conf_valid.max())

    return coverage, mean_conf, std_conf, max_conf


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


def _truncate_graph_to_length(graph: dict, trunc_len: int):
    """
    Truncate node-level graph tensors to trunc_len and drop edges that point beyond trunc_len.

    Expected graph keys from your current pipeline:
      coords, edge_index, edge_attr, edge_weight,
      has_structure, confidence, coverage, mean_confidence, std_confidence, max_confidence,
      is_alphafold, is_experimental, resolution
    """
    out = dict(graph)

    # node-level tensors
    out["coords"] = graph["coords"][:trunc_len]
    out["has_structure"] = graph["has_structure"][:trunc_len]
    out["confidence"] = graph["confidence"][:trunc_len]

    edge_index = graph["edge_index"]
    keep = (edge_index[0] < trunc_len) & (edge_index[1] < trunc_len)

    out["edge_index"] = edge_index[:, keep]
    out["edge_attr"] = graph["edge_attr"][keep]
    out["edge_weight"] = graph["edge_weight"][keep]

    # recompute summary stats after truncation
    coverage, mean_conf, std_conf, max_conf = compute_global_stats(
        out["has_structure"], out["confidence"]
    )
    out["coverage"] = coverage
    out["mean_confidence"] = mean_conf
    out["std_confidence"] = std_conf
    out["max_confidence"] = max_conf

    return out
