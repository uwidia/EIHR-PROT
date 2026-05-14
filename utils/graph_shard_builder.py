from __future__ import annotations

import csv
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import torch
from torch_geometric.nn import radius_graph
from tqdm import tqdm

logger = logging.getLogger(__name__)

ManifestRow = dict[str, Any]
ProteinInfo = dict[str, Any]
GraphDict = dict[str, Any]


def build_aligned_graph_shards(
    manifest_path: str | Path,
    fasta_path: str | Path,
    structure_dir: str | Path,
    output_dir: str | Path,
    get_protein_info_fn: Callable[[Path], Sequence[ProteinInfo]],
    process_single_entry_fn: Callable[..., tuple[str, GraphDict | None]],
    cutoff: float = 10.0,
) -> None:
    """Build graph shards aligned 1:1 with ESM shard and local sequence indices.

    Failed graph builds are saved as None so every graph position still matches
    the corresponding ESM representation index.
    """

    manifest_rows = _load_esm_manifest(manifest_path)
    output_dir = Path(output_dir).resolve()
    structure_dir = Path(structure_dir).resolve()

    label_lookup = _build_label_to_fasta_info(fasta_path, get_protein_info_fn)

    rows_by_shard: defaultdict[int, list[ManifestRow]] = defaultdict(list)
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
        torch.save(metadata, output_dir / "metadata.pt")
        logger.info(f"Done! Graphs built: {total_ok} | Graphs failed: {total_failed} | Output dir: {output_dir}")


def process_single_entry(
    label: str,
    chain_id: str,
    structure_file: str | Path,
    fasta_seq: str,
    cutoff: float = 10,
) -> tuple[str, GraphDict | None]:
    """Run the CIF-to-graph pipeline for one protein entry."""

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


def parse_structure(
    structure_file: str | Path,
    fasta_seq: str,
    chain_id: str,
) -> GraphDict | None:
    """Parse a CIF chain into residue-level structure, confidence, and metadata tensors."""
    structure_file = Path(structure_file)

    try:
        block = _read_cif_block(structure_file)
        table = _find_atom_site_table(block, structure_file)
        if table is None:
            return None

        atom_site = _extract_atom_site_columns(table)
        metadata = _extract_structure_metadata(block)

        chain_data = _filter_chain_atoms(atom_site, chain_id, structure_file)
        if chain_data is None:
            return None

        ca_data = _filter_ca_atoms(chain_data, structure_file)
        if ca_data is None:
            return None


        residue_arrays = _build_full_residue_arrays(ca_data, len(fasta_seq), structure_file)
        if residue_arrays is None:
            return None

        conf = _calculate_confidence(
            residue_arrays["has_structure"],
            residue_arrays["full_b_factor"],
            residue_arrays["full_occupancy"],
            len(fasta_seq),
        ) #dim -> (L, )
        if conf is None:
            logger.error(f"parse_structure | [CONFIDENCE FAIL] {structure_file.stem}")
            return None

        has_confidence = residue_arrays["has_structure"].copy()
        return {
            "coords": torch.from_numpy(residue_arrays["full_coords"]),
            "has_structure": torch.from_numpy(residue_arrays["has_structure"]),
            "confidence": torch.from_numpy(conf.astype(np.float32)),
            "has_confidence": torch.from_numpy(has_confidence),
            "resolution": metadata["resolution"]
        }

    except Exception as e:
        logger.error(f"[EXCEPTION] {structure_file}: {e}")
        return None


def _read_cif_block(structure_file: Path) -> gemmi.cif.Block:
    """Read a CIF file and return its sole data block."""
    doc = gemmi.cif.read_file(str(structure_file))
    return doc.sole_block()


def _find_atom_site_table(
    block: gemmi.cif.Block,
    structure_file: Path,
) -> Any | None:
    """Return the atom-site table required for CA-level graph construction."""
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

    return table


def _extract_atom_site_columns(table: Any) -> dict[str, np.ndarray]:
    """Extract atom-site columns into typed NumPy arrays."""
    atom_name = np.array(table.column(0))
    raw_seq_id = np.array(table.column(1))

    label_seq_id = np.array([_safe_int(x) for x in raw_seq_id], dtype=np.int32) 
    res_name = np.array(table.column(2))
    chain = np.array(table.column(3))

    x = np.array(table.column(4), dtype=np.float32) 
    y = np.array(table.column(5), dtype=np.float32)
    z = np.array(table.column(6), dtype=np.float32)
    coords_all = np.stack([x, y, z], axis=1)

    bfactor = np.array(table.column(7), dtype=np.float32)
    occupancy = np.array(table.column(8), dtype=np.float32)

    return {
        "atom_name": atom_name,
        "label_seq_id": label_seq_id,
        "res_name": res_name,
        "chain": chain,
        "coords_all": coords_all,
        "bfactor": bfactor,
        "occupancy": occupancy,
    }


def _extract_structure_metadata(block: gemmi.cif.Block) -> dict[str, float | str | None]:
    """Extract structure-level metadata used downstream or kept for inspection."""
    # Retrive metadata
    methods = block.find_values("_exptl.method")
    method = "; ".join(methods) if methods else None

    resolution = block.find_value("_refine.ls_d_res_high")
    resolution = float(resolution) if resolution not in [None, "?", "."] else None

    return {
        "method": method,
        "resolution": resolution,
    }


def _filter_chain_atoms(
    atom_site: dict[str, np.ndarray],
    chain_id: str,
    structure_file: Path,
) -> dict[str, np.ndarray] | None:
    """Keep atoms from the requested author chain."""
    # Chain filtering for experimentally-derived structures (i.e. PDB only)
    mask_chain = (atom_site["chain"] == chain_id)
    if mask_chain.sum() == 0:
        logger.error(f"parse_structure | [CHAIN FAIL] {structure_file.stem}: chain {chain_id}")
        return None

    return {
        "atom_name": atom_site["atom_name"][mask_chain],
        "label_seq_id": atom_site["label_seq_id"][mask_chain],
        "res_name": atom_site["res_name"][mask_chain],
        "coords_all": atom_site["coords_all"][mask_chain],
        "bfactor": atom_site["bfactor"][mask_chain],
        "occupancy": atom_site["occupancy"][mask_chain],
    }


def _filter_ca_atoms(
    atom_site: dict[str, np.ndarray],
    structure_file: Path,
) -> dict[str, np.ndarray] | None:
    """Keep alpha-carbon atoms and their residue-level fields."""
    # filter for alpha carbon (CA) atoms only
    ca_mask = (atom_site["atom_name"] == "CA")

    if ca_mask.sum() == 0:
        logger.error(f"parse_structure | [NO CA] {structure_file.stem}")
        return None

    return {
        "atom_name": atom_site["atom_name"][ca_mask],
        "label_seq_id": atom_site["label_seq_id"][ca_mask],
        "res_name": atom_site["res_name"][ca_mask],
        "coords": atom_site["coords_all"][ca_mask],
        "b_vals": atom_site["bfactor"][ca_mask],
        "occ_vals": atom_site["occupancy"][ca_mask],
    }


def _build_full_residue_arrays(
    ca_data: dict[str, np.ndarray],
    sequence_length: int,
    structure_file: Path,
) -> dict[str, np.ndarray] | None:
    """Map CA records into full-length residue arrays indexed by FASTA position."""
    L = sequence_length
    idx = ca_data["label_seq_id"] - 1 #zero-indexing

    # Bounds check (i.e. Return none if no sequence information is present in the fasta file)
    valid = (idx >= 0) & (idx < L)

    if valid.sum() == 0:
        logger.error(f"parse_structure | [INDEX FAIL] {structure_file.stem}")
        return None

    idx = idx[valid]
    coords = ca_data["coords"][valid]
    b_vals = ca_data["b_vals"][valid]
    occ_vals = ca_data["occ_vals"][valid]

    # Build full arrays for graph object
    full_coords = np.zeros((L, 3), dtype=np.float32)
    full_b_factor = np.zeros(L, dtype=np.float32)
    full_occupancy = np.zeros(L, dtype=np.float32)
    has_structure = np.zeros(L, dtype=bool)

    full_coords[idx] = coords
    full_b_factor[idx] = b_vals
    full_occupancy[idx] = occ_vals
    has_structure[idx] = True

    return {
        "full_coords": full_coords,
        "full_b_factor": full_b_factor,
        "full_occupancy": full_occupancy,
        "has_structure": has_structure,
    }


def _safe_int(x: Any) -> int:
    """Convert CIF sequence IDs to integers, using -1 for invalid values."""
    try:
        return int(x)
    except:
        return -1

def construct_graph(
    data: GraphDict,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> GraphDict:
    """Assemble the final graph dictionary and derived graph-level fields."""

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
        "resolution": resolution,                 # float or None      
    }

    return graph


def build_radius_graph(
    coords: torch.Tensor,
    has_structure: torch.Tensor,
    cutoff: float = 10,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build a CA-only radius graph with distance and direction edge features."""

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


def _load_esm_manifest(manifest_path: str | Path) -> list[ManifestRow]:
    """Read and validate the ESM manifest rows in manifest order."""
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


def _build_label_to_fasta_info(
    fasta_path: str | Path,
    get_protein_info_fn: Callable[[Path], Sequence[ProteinInfo]],
) -> dict[str, ProteinInfo]:
    """Build a manifest-label lookup for FASTA sequence and structure identifiers."""
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
    shard_records: list[GraphDict | None],
    shard_id: int,
    output_dir: Path,
) -> None:
    """Save one graph shard, preserving local index alignment with None placeholders."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"graph_shard_{shard_id:04d}.pt"
    torch.save(
        {
            "graphs": shard_records,
        },
        path,
    )


def _calculate_confidence(
    has_structure: np.ndarray,
    full_b_factor: np.ndarray,
    full_occupancy: np.ndarray,
    sequence_length: int,
) -> np.ndarray | None:
    """Convert B-factors and occupancy into per-residue confidence scores."""
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


def _compute_global_stats(
    has_structure: torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[float, float, float]:
    """Compute coverage and confidence summary statistics."""
    mask = has_structure.bool()

    if mask.sum() == 0:
        return 0.0, 0.0, 0.0

    conf_valid = confidence[mask]

    coverage = float(mask.float().mean())
    mean_conf = float(conf_valid.mean())
    std_conf = float(conf_valid.std())

    return coverage, mean_conf, std_conf


def _compute_edge_weights(
    edge_index: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """Compute each edge weight as the product of endpoint confidence scores."""
    i, j = edge_index
    return confidence[i] * confidence[j]
