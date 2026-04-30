from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Mapping
import json
from goatools.obo_parser import GODag
import torch


GO_ASPECT_TO_COLUMN = {
    "MF": "GO-terms (molecular_function)",
    "BP": "GO-terms (biological_process)",
    "CC": "GO-terms (cellular_component)",
}

#Standard root terms for each GO aspect
ROOT_TERMS = {
    "MF": "GO:0003674",
    "BP": "GO:0008150",
    "CC": "GO:0005575",
}



def build_go_annotations_list(
    tsv_path: str | Path,
    obo_path: str | Path,
    go_aspect: str,
    *,
    keep_ids: Iterable[str] | None = None,
    remove_root_term: bool = True,
    min_term_freq: int | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Build:
        label_to_go_terms: dict[str, list[str]]
        go_terms: list[str]

    Workflow:
      1) parse direct terms from TSV for a chosen go_aspect
      2) propagate using GOATOOLS
      3) optionally remove ontology root term
      4) optionally filter rare terms by training frequency
      5) return sorted per-protein terms and sorted global vocabulary
    """
    go_aspect = go_aspect.upper()
    if go_aspect not in {"BP", "MF", "CC"}:
        raise ValueError("go_aspect must be one of: 'BP', 'MF', 'CC'")

    go_dag = GODag(str(obo_path), optional_attrs={"relationship"})

    # Parse direct annotations from TSV
    label_to_direct = _extract_go_terms(
        tsv_path=tsv_path,
        go_aspect=go_aspect,
        keep_ids=keep_ids,
    )

    # Propagate GO terms
    label_to_propagated: dict[str, set[str]] = {}
    for protein_id, direct_terms in label_to_direct.items():
        propagated = _propagate_terms_with_goatools(direct_terms, go_dag)

        # Keep only terms that truly belong to the requested go_aspect
        filtered = {
            go_id
            for go_id in propagated
            if go_id in go_dag and go_dag[go_id].namespace == {
                "BP": "biological_process",
                "MF": "molecular_function",
                "CC": "cellular_component",
            }[go_aspect]
        }

        if remove_root_term:
            filtered.discard(ROOT_TERMS[go_aspect])

        label_to_propagated[protein_id] = filtered

    # Frequency filter
    if min_term_freq is not None and min_term_freq > 1:
        term_counts: dict[str, int] = {}
        for terms in label_to_propagated.values():
            for go_id in terms:
                term_counts[go_id] = term_counts.get(go_id, 0) + 1

        keep_terms = {go_id for go_id, c in term_counts.items() if c >= min_term_freq}

        for protein_id in label_to_propagated:
            label_to_propagated[protein_id] &= keep_terms

    # Sort per protein and build sorted vocabulary
    label_to_go_terms = {
        protein_id: sorted(terms)
        for protein_id, terms in label_to_propagated.items()
    }

    go_terms = sorted({
        go_id
        for terms in label_to_go_terms.values()
        for go_id in terms
    })

    return label_to_go_terms, go_terms

def build_subject_go_index(
    label_to_go_terms: Mapping[str, Sequence[str]],
    go_term_to_idx: Mapping[str, int],
) -> dict[str, list[int]]:
    """
    Convert label -> GO term strings into label -> sorted GO index lists.
    """
    subject_to_indices: dict[str, list[int]] = {}

    for label, go_terms in label_to_go_terms.items():
        indices = sorted(
            {
                go_term_to_idx[go_term]
                for go_term in go_terms
                if go_term in go_term_to_idx
            }
        )
        subject_to_indices[label] = indices

    return subject_to_indices


def save_subject_go_index(
    subject_to_indices: Mapping[str, Sequence[int]],
    output_json_path: str | Path,
) -> Path:
    output_json_path = Path(output_json_path).resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: list(value) for key, value in subject_to_indices.items()}
    output_json_path.write_text(json.dumps(serializable, indent=2, sort_keys=True))
    return output_json_path

def save_go_vocab(go_terms: Sequence[str], output_json_path: str | Path) -> Path:
    output_json_path = Path(output_json_path).resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(list(go_terms), indent=2))
    return output_json_path

def build_child_parent_idx_pairs(
    obo_path: str | Path,
    go_terms: list[str],
) -> torch.Tensor:
    """
    Build direct child-parent GO index pairs for one aspect-specific GO vocab.

    Returns:
        LongTensor of shape (num_pairs, 2)
        column 0 = child_idx
        column 1 = parent_idx
    """
    go_dag = GODag(str(obo_path), optional_attrs={"relationship"})
    go_to_idx = {go: i for i, go in enumerate(go_terms)}

    pairs = []

    for child_go in go_terms:
        if child_go not in go_dag:
            continue

        child_idx = go_to_idx[child_go]

        for parent in go_dag[child_go].parents:
            parent_go = parent.id

            if parent_go in go_to_idx:
                parent_idx = go_to_idx[parent_go]
                pairs.append((child_idx, parent_idx))

    return torch.tensor(pairs, dtype=torch.long)

def _extract_go_terms(
    tsv_path: str | Path,
    go_aspect: str,
    keep_ids: Iterable[str] | None = None,
) -> dict[str, set[str]]:
    """
    Parse TSV and return:
        protein_id -> direct GO terms for one go_aspect only

    The TSV contains a header line in the format:
    ### PDB-chain    GO-terms (molecular_function)    GO-terms (biological_process)    GO-terms (cellular_component)
    """
    go_aspect = go_aspect.upper()
    if go_aspect not in GO_ASPECT_TO_COLUMN:
        raise ValueError("go_aspect must be one of: 'BP', 'MF', 'CC'")

    keep_ids_set = set(keep_ids) if keep_ids is not None else None
    target_col = GO_ASPECT_TO_COLUMN[go_aspect]

    header: list[str] | None = None
    in_table = False
    label_to_terms: dict[str, set[str]] = {}

    tsv_path = Path(tsv_path)

    with tsv_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            if line.startswith("### PDB-chain\t"):
                header = line[4:].split("\t")  # strip leading "### "
                in_table = True
                continue

            if not in_table or not line.strip():
                continue

            if header is None:
                raise RuntimeError("Failed to detect TSV annotation table header.")

            fields = line.split("\t")
            if len(fields) < len(header):
                fields += [""] * (len(header) - len(fields))

            row = dict(zip(header, fields))
            protein_id = row["PDB-chain"]

            if keep_ids_set is not None and protein_id not in keep_ids_set:
                continue

            raw_terms = row.get(target_col, "").strip()
            direct_terms = {term.strip() for term in raw_terms.split(",") if term.strip()}
            label_to_terms[protein_id] = direct_terms

    return label_to_terms


def _propagate_terms_with_goatools(
    direct_terms: set[str],
    go_dag: GODag,
    *,
    include_term_itself: bool = True,
) -> set[str]:
    """
    Propagate a set of direct GO terms to all ancestors using GOATOOLS.
    Obsolete/missing terms are skipped.
    """
    propagated: set[str] = set()

    for go_id in direct_terms:
        if go_id not in go_dag:
            continue

        term_obj = go_dag[go_id]
        ancestors = term_obj.get_all_parents()

        if include_term_itself:
            propagated.add(go_id)
        propagated.update(ancestors)

    return propagated
