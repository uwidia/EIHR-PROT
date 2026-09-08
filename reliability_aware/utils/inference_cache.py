"""Full prediction cache export shared by existing inference routes."""
from __future__ import annotations

import argparse
import logging
import numpy as np
from reliability_aware.utils.prediction_cache import PredictionCache, sha256_file, write_prediction_cache

LOGGER = logging.getLogger('run_inference')


def _write_prediction_cache_if_requested(*, args: argparse.Namespace, checkpoint: dict | None,
                                         go_terms: list[str], results: dict,
                                         train_annotations: list[set[str]]) -> None:
    """Persist one lossless cache after a labeled inference pass."""
    if args.mode != "evaluate" or args.prediction_cache_path is None:
        return
    sources = {"test_fasta": args.test_fasta, "go_vocab": args.go_vocab_path,
               "obo": args.obo_path, "annotations": args.go_annotation_path,
               "checkpoint": args.checkpoint}
    if args.ablation in {'sequence_homology_fixed_fusion', 'sequence_homology_identity_fusion'}:
        sources.update({'identity_sidecar': args.identity_sidecar_path,
                        'manifest': args.test_manifest_path, 'train_fasta': args.train_fasta,
                        'train_annotations': args.train_go_annotation_path or args.go_annotation_path})
    source_hashes = {name: {"path": str(path), "sha256": sha256_file(path)}
                     for name, path in sources.items() if path is not None and path.exists()}
    cache = PredictionCache(
        protein_ids=np.asarray(results["labels"], dtype=str),
        go_terms=np.asarray(go_terms, dtype=str),
        probabilities=results["y_prob"].detach().cpu().numpy(),
        labels=results["y_true"].detach().cpu().numpy(),
        eligibility=np.ones(len(results["labels"]), dtype=bool),
        gate_weights=(results["gate_weights"].detach().cpu().numpy()
                      if "gate_weights" in results else None),
        metadata={"dataset": getattr(args, "dataset_id", "PDB"), "model_id": args.ablation,
                  "go_aspect": args.go_aspect, "coverage": "annotation-eligible-only",
                  "checkpoint_path": str(args.checkpoint) if args.checkpoint else None,
                  "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
                  "checkpoint_hparams": checkpoint.get("hparams") if checkpoint else None,
                  "source_hashes": source_hashes,
                  "train_annotations": [sorted(terms) for terms in train_annotations],
                  "numerical_settings": {"probability_dtype": str(results["y_prob"].dtype),
                                         "model_mode": "eval; gradients disabled"}},
    )
    if args.ablation in {'sequence_homology_fixed_fusion', 'sequence_homology_identity_fusion'}:
        from reliability_aware.utils.fusion_protocol import resource_fingerprint
        cache.metadata.update({'fusion_parameters': checkpoint['fusion_parameters'],
                               'fusion_protocol_version': checkpoint['fusion_protocol_version'],
                               'validation_exclude_ids': checkpoint['validation_exclude_ids'],
                               'policy': checkpoint['adaptation'],
                               'input_shards': {k: resource_fingerprint(p) for k, p in
                                   [('esm', args.test_esm_shard_dir), ('homology', args.test_homology_shard_dir)]}})
    written = write_prediction_cache(cache, args.prediction_cache_path,
                                     overwrite=args.overwrite_prediction_cache)
    LOGGER.info("Saved validated full-output prediction cache: %s", written)
