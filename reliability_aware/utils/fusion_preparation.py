"""Audited exclusion and retained-alignment matching, without changing old evidence."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

from reliability_aware.utils.diamond_homology import (
    DIAMOND_OUTFMT_FIELDS, DiamondSearchConfig, _parse_diamond_hits,
    _retain_valid_hits, read_fasta_as_dict,
)
from reliability_aware.utils.identity_fusion import build_identity_sidecar
from reliability_aware.utils.prediction_cache import canonical_ids_hash, sha256_file


def write_once(path, contents):
    path = Path(path)
    if path.exists():
        if path.read_text() != contents:
            raise ValueError(f"Existing prepared artifact differs: {path}; use a new prepared directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def alignment_fingerprint(hit):
    return tuple(getattr(hit, k) for k in (
        'qseqid', 'sseqid', 'evalue', 'bitscore', 'qcov', 'qlen', 'slen', 'length', 'pident'))


def filtered_hits(source, target, excluded, *, enriched):
    """Exclude whole queries BEFORE strict parsing; audit every removed raw row."""
    lines, removed, counts = [], [], {}
    with Path(source).open(newline='') as handle:
        for number, row in enumerate(csv.reader(handle, delimiter='\t'), 1):
            if not row:
                continue
            if row[0] in excluded:
                removed.append({'line': number, 'query_id': row[0], 'fields': row})
                continue
            if len(row) not in ((10,) if enriched else (9, 10)):
                raise ValueError(f'{source}: line {number}: expected {10 if enriched else "9 or 10"} fields')
            try:
                nums = [float(x) for x in row[2:9]]
                if not all(math.isfinite(x) for x in nums):
                    raise ValueError('nonfinite alignment field')
                if nums[0] < 0 or nums[1] <= 0 or not 0 <= nums[2] <= 100 or not 0 <= nums[6] <= 100:
                    raise ValueError('invalid evalue, bitscore, coverage, or pident')
                if int(row[7]) <= 0:
                    raise ValueError('alignment length must be positive')
                if enriched:
                    count = int(row[9])
                    fp = (row[0], row[1], *nums)
                    if fp in counts and counts[fp] != count:
                        raise ValueError('ambiguous alignment fingerprint with conflicting nident')
                    counts[fp] = count
            except ValueError as exc:
                raise ValueError(f'{source}: line {number}: {exc}') from exc
            lines.append('\t'.join(row) + '\n')
    write_once(target, ''.join(lines))
    return removed


def prepare_identity_split(*, resources, split):
    item = resources[split]
    root = Path(resources['prepared_dir'])
    root.mkdir(parents=True, exist_ok=True)
    excluded = set(resources['validation_exclude_ids']) if split == 'pdb_val' else set()
    queries = read_fasta_as_dict(item['fasta'])
    if excluded - set(queries):
        raise ValueError(f'Validation exclusion IDs absent from FASTA: {sorted(excluded - set(queries))}')
    kept = [q for q in queries if q not in excluded]
    write_once(root / f'{split}.ids.txt', ''.join(q + '\n' for q in kept))
    write_once(root / f'{split}.fasta', ''.join(f'>{q}\n{queries[q]}\n' for q in kept))
    original = root / f'{split}_original_hits.tsv'
    enriched = root / f'{split}_hits_with_nident.tsv'
    report_path = root / f'{split}_evidence_report.json'
    report = {'split': split, 'status': 'failed', 'excluded_query_ids': sorted(excluded),
              'original_query_count': len(queries), 'remaining_query_count': len(kept),
              'query_ids_sha256': canonical_ids_hash(sorted(kept)),
              'source_hashes': {k: {'path': item[k], 'sha256': sha256_file(Path(item[k]))}
                                for k in ('fasta', 'original_hits', 'enriched_hits')},
              'comparison': 'Exact numeric equality of all nine serialized fields after retention; no tolerance or PID-derived nident',
              'limitation': 'Legacy nine-column evidence has no alignment coordinates; matching validates available fields, not a unique alignment path.'}
    try:
        report['removed_original_rows'] = filtered_hits(item['original_hits'], original, excluded, enriched=False)
        report['removed_enriched_rows'] = filtered_hits(item['enriched_hits'], enriched, excluded, enriched=True)
        old, new = _parse_diamond_hits(original), _parse_diamond_hits(enriched)
        if (set(old) | set(new)) - set(kept):
            raise ValueError('Hits contain query IDs outside the configured FASTA')
        cfg = DiamondSearchConfig(evalue_max=1e-5, min_query_coverage=.30, top_k=10)
        self_hits = split == 'pdb_train'
        mismatches = []
        for q in kept:
            a = _retain_valid_hits(old.get(q, []), cfg, exclude_self=self_hits, query_id=q)
            b = _retain_valid_hits(new.get(q, []), cfg, exclude_self=self_hits, query_id=q)
            if [alignment_fingerprint(h) for h in a] != [alignment_fingerprint(h) for h in b]:
                mismatches.append(q)
        report['mismatched_queries'] = mismatches
        if mismatches:
            raise ValueError(f'{len(mismatches)} queries have changed retained evidence; see {report_path}')
        report['status'] = 'matched'
        report['retention'] = {'evalue_max': cfg.evalue_max, 'min_query_coverage': cfg.min_query_coverage, 'top_k': cfg.top_k, 'exclude_self_hits': self_hits}
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        output = Path(item['identity'])
        # Reuse only a sidecar produced from these exact inputs and this report.
        if output.exists():
            current = json.loads(output.read_text())
            if (current.get('source_hits_sha256') != sha256_file(enriched)
                    or current.get('evidence_report_sha256') != sha256_file(report_path)):
                raise ValueError(f'Stale sidecar: {output}; use a new prepared directory')
            from models.identity_fusion_data import load_identity_sidecar
            load_identity_sidecar(output)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = build_identity_sidecar(query_ids=kept, hits_tsv=enriched, output_path=output,
                                             config=cfg, exclude_self_hits=self_hits)
            payload.update({'evidence_report': str(report_path.resolve()),
                            'evidence_report_sha256': sha256_file(report_path),
                            'query_ids_sha256': canonical_ids_hash(sorted(kept)),
                            'excluded_query_ids': sorted(excluded), 'dataset_split': split,
                            'retention': report['retention'],
                            'schema_fields': DIAMOND_OUTFMT_FIELDS,
                            'audit_sha256': sha256_file(Path(payload['audit_path']))})
            output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        raise
    return report


def verify_frozen_priors(resources, split, aspect):
    """Check existing shard values against original evidence without rewriting them."""
    import torch
    from reliability_aware.utils.diamond_homology import _build_prior_from_hits, _load_subject_go_index
    from reliability_aware.utils.fusion_protocol import resource_fingerprint
    item = resources[split]
    directory = Path(item['homology_shards'].format(aspect=aspect))
    files = sorted(directory.glob('homology_shard_*.pt'))
    if not files:
        raise FileNotFoundError(f'No homology shards: {directory}')
    source = Path(resources['prepared_dir']) / f'{split}_original_hits.tsv'
    excluded = set(resources['validation_exclude_ids']) if split == 'pdb_val' else set()
    filtered_hits(item['original_hits'], source, excluded, enriched=False)
    hits = _parse_diamond_hits(source)
    vocab = Path(resources['go_vocab'].format(aspect=aspect))
    index_path = Path(resources['subject_go_index'].format(aspect=aspect))
    index = _load_subject_go_index(index_path)
    size = len(json.loads(vocab.read_text()))
    cfg = DiamondSearchConfig()
    excluded = set(resources['validation_exclude_ids']) if split == 'pdb_val' else set()
    with Path(item['manifest']).open(newline='') as handle:
        manifest = list(csv.DictReader(handle))
    expected = {r['label']: (int(r['shard_number']), int(r['local_seq_idx'])) for r in manifest if r['label'] not in excluded}
    seen = set()
    for file in files:
        shard_id = int(file.stem.split('_')[-1])
        shard = torch.load(file, map_location='cpu', weights_only=False)
        for local, q in enumerate(shard['labels']):
            if q in excluded:
                continue
            if q in seen or expected.get(q) != (shard_id, local):
                raise ValueError(f'{file}: label/manifest alignment mismatch for {q}')
            retained = _retain_valid_hits(hits.get(q, []), cfg, exclude_self=split == 'pdb_train', query_id=q)
            prior, stats, debug = _build_prior_from_hits(retained, index, size)
            actual = shard['priors'][local]
            if not torch.equal(prior.to(actual.dtype), actual):
                raise ValueError(f'{file}: {q}: frozen prior differs from original retained evidence')
            if not torch.equal(stats.as_gate_features().to(shard['gate_features'][local]), shard['gate_features'][local]):
                raise ValueError(f'{file}: {q}: frozen retrieval features differ from retained evidence')
            seen.add(q)
    if seen != set(expected):
        raise ValueError(f'{directory}: incomplete manifest coverage')
    report = {'status': 'matched', 'split': split, 'aspect': aspect, 'checked_n': len(seen),
              'excluded_query_ids': sorted(excluded),
              'sources': {k: resource_fingerprint(v) for k, v in
                  [('shards', directory), ('manifest', item['manifest']), ('original_hits', item['original_hits']),
                   ('prepared_hits', source), ('vocab', vocab), ('subject_go_index', index_path)]}}
    path = Path(resources['prepared_dir']) / f'{split}_{aspect}_prior_report.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
