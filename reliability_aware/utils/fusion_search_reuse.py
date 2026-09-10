"""Reuse completed fusion searches after verified resource relocation.

Saved protocols and results remain immutable. Only storage locations and
regenerated identity provenance may differ; all model inputs must agree.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from reliability_aware.utils.prediction_cache import sha256_file

_LOCATION_CONFIG_KEYS = {
    'base_dir_search', 'base_dir_final', 'search_results_path', 'resources',
    'train_identity_sidecar_path', 'val_identity_sidecar_path',
}
_IDENTITY_PROVENANCE_KEYS = {'evidence_report', 'evidence_report_sha256', 'audit_path'}


def search_results_path(config, aspect, search_dir=None):
    if search_dir is not None:
        return Path(search_dir) / aspect / 'search_results.json'
    return Path(config.get('search_results_path') or
                Path(config['base_dir_search']) / aspect / 'search_results.json')


def _content(fingerprint):
    """A location never substitutes for an actual content fingerprint."""
    keys = set(fingerprint)
    if keys not in ({'path', 'sha256'}, {'path', 'files'}):
        raise ValueError('Search resource has no supported content fingerprint')
    return {k: v for k, v in fingerprint.items() if k != 'path'}


def _equivalent_identity(saved, current):
    from models.identity_fusion_data import load_identity_sidecar

    old_path, new_path = Path(saved['path']), Path(current['path'])
    # The historical sidecar itself must still match the completed search.
    # Its old report may have moved/been replaced; current evidence is validated
    # independently, and every non-provenance sidecar field must be identical.
    if sha256_file(old_path) != saved['sha256']:
        raise ValueError(f'Historical identity sidecar differs from search: {old_path}')
    if sha256_file(new_path) != current['sha256']:
        raise ValueError(f'Current identity sidecar changed: {new_path}')
    load_identity_sidecar(new_path)
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    if not old.get('evidence_report') or not new.get('evidence_report'):
        raise ValueError('Search reuse requires audited identity sidecars')
    strip = lambda value: {k: v for k, v in value.items() if k not in _IDENTITY_PROVENANCE_KEYS}
    if strip(old) != strip(new):
        raise ValueError('Identity records, retained-hit evidence, or policy changed since search')
    return {
        'historical_sidecar': saved, 'current_sidecar': current,
        'comparison': 'Exact equality except evidence-report location/hash and audit location; '
                      'current evidence and audit independently validated',
    }


def _equivalent_homology(saved, current):
    """The builder summary is not a model input; every tensor file must match."""
    old, new = saved.get('files', {}), current.get('files', {})
    summary = 'homology_shard_metadata.json'
    if set(old) != set(new) or summary not in old:
        raise ValueError('Homology shard file set changed since search')
    tensors = [name for name in old if name.startswith('homology_shard_') and name.endswith('.pt')]
    if not tensors or any(old[name] != new[name] for name in old if name != summary):
        raise ValueError('Homology shard tensor contents changed since search')
    return {'saved': saved, 'current': current, 'tensor_file_count': len(tensors),
            'comparison': 'All shard tensor files byte-identical; only builder summary metadata differs'}


def validate_relocated_protocol(saved, current):
    blocks = {'configuration', 'resources', 'reference_source_hashes'}
    if {k: v for k, v in saved.items() if k not in blocks} != {
            k: v for k, v in current.items() if k not in blocks}:
        raise ValueError('Search cohort, vocabulary, model, seed, or protocol changed')
    for value in (saved, current):
        if not blocks <= set(value):
            raise ValueError('Search protocol is missing required provenance')
    old_config, new_config = saved['configuration'], current['configuration']
    if {k: v for k, v in old_config.items() if k not in _LOCATION_CONFIG_KEYS} != {
            k: v for k, v in new_config.items() if k not in _LOCATION_CONFIG_KEYS}:
        raise ValueError('Search/training settings changed; only resource and output paths may differ')
    report = {'configuration_path_changes': {
        k: {'saved': old_config.get(k), 'current': new_config.get(k)}
        for k in sorted(_LOCATION_CONFIG_KEYS) if old_config.get(k) != new_config.get(k)},
        'relocated_resources': {}, 'equivalent_identity_sidecars': {}, 'equivalent_homology_shards': {}}
    for block in ('resources', 'reference_source_hashes'):
        old, new = saved[block], current[block]
        if set(old) != set(new):
            raise ValueError(f'Search {block} set changed')
        for name, fingerprint in old.items():
            actual = new[name]
            if _content(fingerprint) != _content(actual):
                if block == 'resources' and name in ('pdb_train/identity', 'pdb_val/identity'):
                    report['equivalent_identity_sidecars'][name] = _equivalent_identity(fingerprint, actual)
                elif block == 'resources' and name in ('pdb_train/homology_shards', 'pdb_val/homology_shards'):
                    report['equivalent_homology_shards'][name] = _equivalent_homology(fingerprint, actual)
                else:
                    raise ValueError(f'Search resource contents changed: {block}/{name}')
            elif fingerprint['path'] != actual['path']:
                report['relocated_resources'][f'{block}/{name}'] = {
                    'saved': fingerprint['path'], 'current': actual['path'],
                    'content': _content(actual)}
    return report


def prepare_final_candidates(path, count, metadata, *, allow_relocated=False):
    """Validate the selected search and return candidates plus immutable provenance."""
    path = Path(path)
    protocol = path.parent / 'protocol.json'
    saved = json.loads(protocol.read_text())
    current = json.loads(json.dumps(metadata))
    comparison = {'mode': 'exact_protocol_match'}
    if saved != current:
        if not allow_relocated:
            raise ValueError('Search results use a different cohort, resources, model, or configuration. '
                             'For equivalent relocated resources, use final --search-dir with the original search directory.')
        comparison = {'mode': 'verified_resource_relocation', **validate_relocated_protocol(saved, current)}
    records = json.loads(path.read_text())
    if count <= 0 or not records:
        raise ValueError('Need positive top_k_params and nonempty search results')
    if any(not math.isfinite(float(row['score'])) for row in records):
        raise ValueError('Nonfinite search score')
    selected = sorted(records, key=lambda row: row['score'], reverse=True)[:count]
    report = {'search_results': {'path': str(path), 'sha256': sha256_file(path)},
              'search_protocol': {'path': str(protocol), 'sha256': sha256_file(protocol)},
              'comparison': comparison, 'selected_scores': [row['score'] for row in selected],
              'selected_hparams': [row['hparams'] for row in selected]}
    return [row['hparams'] for row in selected], report
