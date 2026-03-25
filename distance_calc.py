import gemmi
import numpy as np
import torch

def parse_structure(cif_path, chain_id=None, eps=1e-6):
    try:
        doc = gemmi.cif.read_file(cif_path)
        block = doc.sole_block()

        atom_site = block.find_loop("_atom_site.")

        # --- Extract columns (vectorized) ---
        atom_name = np.array(atom_site.get_column("_atom_site.label_atom_id"))
        res_id    = np.array(atom_site.get_column("_atom_site.label_seq_id"), dtype=np.int32)
        chain     = np.array(atom_site.get_column("_atom_site.auth_asym_id"))

        x = np.array(atom_site.get_column("_atom_site.Cartn_x"), dtype=np.float32)
        y = np.array(atom_site.get_column("_atom_site.Cartn_y"), dtype=np.float32)
        z = np.array(atom_site.get_column("_atom_site.Cartn_z"), dtype=np.float32)

        bfactor   = np.array(atom_site.get_column("_atom_site.B_iso_or_equiv"), dtype=np.float32)
        occupancy = np.array(atom_site.get_column("_atom_site.occupancy"), dtype=np.float32)

        coords = np.stack([x, y, z], axis=1)

        # --- Method detection ---
        method = block.find_value("_exptl.method")
        is_alphafold = (method is None)

        # --- Resolution ---
        resolution = block.find_value("_refine.ls_d_res_high")
        resolution = float(resolution) if resolution not in [None, "?", "."] else None

        # --- Chain filtering ---
        if chain_id is not None:
            mask_chain = (chain == chain_id)
        else:
            mask_chain = np.ones(len(chain), dtype=bool)

        # --- Cα selection ---
        ca_mask = (atom_name == "CA") & mask_chain

        if ca_mask.sum() == 0:
            return None

        res_ids = res_id[ca_mask]
        coords  = coords[ca_mask]
        bfactor = bfactor[ca_mask]
        occupancy = occupancy[ca_mask]

        # --- Sort residues ---
        order = np.argsort(res_ids)
        res_ids = res_ids[order]
        coords  = coords[order]
        bfactor = bfactor[order]
        occupancy = occupancy[order]

        # --- Build full-length arrays ---
        L = res_ids.max()

        full_coords = np.zeros((L, 3), dtype=np.float32)
        has_structure = np.zeros(L, dtype=bool)

        full_b = np.zeros(L, dtype=np.float32)
        full_occ = np.zeros(L, dtype=np.float32)

        idx = res_ids - 1
        full_coords[idx] = coords
        full_b[idx] = bfactor
        full_occ[idx] = occupancy
        has_structure[idx] = True

        # --- Confidence computation ---
        has_conf = has_structure.copy()

        if is_alphafold:
            # AF: B-factor = pLDDT
            conf = full_b / 100.0

        else:
            valid_b = full_b[has_structure]

            if len(valid_b) == 0:
                return None

            median = np.median(valid_b)
            mad = np.median(np.abs(valid_b - median)) + eps

            z = (full_b - median) / mad

            local_conf = full_occ * (1 / (1 + np.exp(z)))  # sigmoid(-z)

            # --- Global resolution scaling ---
            if resolution is not None:
                global_conf = 1.0 / (1.0 + max(0.0, resolution - 2.5))
            else:
                global_conf = 1.0  # fallback

            conf = global_conf * local_conf

        return {
            "coords": torch.from_numpy(full_coords),
            "has_structure": torch.from_numpy(has_structure),
            "confidence": torch.from_numpy(conf.astype(np.float32)),
            "has_confidence": torch.from_numpy(has_conf),
            "resolution": resolution,
            "is_alphafold": is_alphafold
        }

    except Exception:
        return None  # failure cache handled outside

def build_edges(coords, has_structure, cutoff=10.0):
    coords = coords.numpy()
    mask = has_structure.numpy()

    valid_idx = np.where(mask)[0]
    valid_coords = coords[valid_idx]

    if len(valid_coords) == 0:
        return None

    diff = valid_coords[:, None, :] - valid_coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    edge_mask = (dist < cutoff) & (dist > 0)

    src, dst = np.where(edge_mask)

    edge_index = np.stack([valid_idx[src], valid_idx[dst]], axis=0)

    return torch.from_numpy(edge_index).long()

#This is the actual function that you'll call in your ESM loop
def process_structure(cif_path):
    data = parse_structure(cif_path)

    if data is None:
        return None

    edge_index = build_edges(
        data["coords"],
        data["has_structure"]
    )

    if edge_index is None:
        return None

    data["edge_index"] = edge_index

    return data

#Assuming i want to run it with paralle processing inside the ESM
from multiprocessing import Pool

def process_many(cif_paths, num_workers=8):
    with Pool(num_workers) as p:
        results = p.map(process_structure, cif_paths)

    return [r for r in results if r is not None]

#Example of how i could possibly use this inside the ESM code
from structure_parser import process_structure

for protein in batch:
    struct_data = process_structure(cif_path)

    if struct_data is None:
        continue

    shard_buffer.add(
        embedding,
        struct_data["coords"],
        struct_data["edge_index"],
        struct_data["confidence"],
        struct_data["has_structure"],
        struct_data["has_confidence"]
    )
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