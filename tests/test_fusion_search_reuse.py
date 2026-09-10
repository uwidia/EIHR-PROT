"""Existing searches may follow relocated evidence, never changed model inputs."""
import copy
import json
from pathlib import Path

import pytest

from reliability_aware.utils.fusion_preparation import prepare_identity_split
from reliability_aware.utils.fusion_protocol import resource_fingerprint
from reliability_aware.utils.fusion_search_reuse import prepare_final_candidates, search_results_path
from scripts import fusion_baselines


@pytest.fixture
def handoff(tmp_path):
    fasta = tmp_path / 'train.fasta'
    fasta.write_text('>q\n' + 'A' * 100 + '\n')
    hits = tmp_path / 'hits.tsv'
    hits.write_text('q\ts\t1e-9\t100\t50\t100\t100\t50\t50\t25\n')
    root = {'validation_exclude_ids': [], 'pdb_train': {
        'fasta': str(fasta), 'original_hits': str(hits), 'enriched_hits': str(hits)}}
    versions = []
    for name in ('old', 'new'):
        resources = copy.deepcopy(root)
        resources['prepared_dir'] = str(tmp_path / name)
        resources['pdb_train']['identity'] = str(tmp_path / name / 'identity.json')
        prepare_identity_split(resources=resources, split='pdb_train')
        versions.append(resources)
    old, new = versions
    reference = tmp_path / 'reference.py'
    reference.write_text('# reference\n')
    moved = tmp_path / 'relocated_reference.py'
    moved.write_bytes(reference.read_bytes())
    metadata = {'fusion_protocol_version': 'validation_exclude_two_v1',
                'model_type': 'sequence_homology_identity_fusion', 'go_aspect': 'BP',
                'go_terms': ['g1'], 'train_ids_sha256': 'cohort', 'seed': 42,
                'validation_exclude_ids': ['2VAU-A', '5LSQ-A'],
                'configuration': {'seed': 42, 'final_epochs': 110, 'top_k_params': 1,
                                  'base_dir_search': 'old/search', 'base_dir_final': 'old/final',
                                  'train_identity_sidecar_path': old['pdb_train']['identity']},
                'resources': {'pdb_train/fasta': resource_fingerprint(fasta),
                              'pdb_train/identity': resource_fingerprint(old['pdb_train']['identity'])},
                'reference_source_hashes': {'predict.py': resource_fingerprint(reference)}}
    current = copy.deepcopy(metadata)
    current['configuration'].update(base_dir_search='new/search', base_dir_final='new/final',
                                    train_identity_sidecar_path=new['pdb_train']['identity'])
    current['resources']['pdb_train/identity'] = resource_fingerprint(new['pdb_train']['identity'])
    current['reference_source_hashes']['predict.py'] = resource_fingerprint(moved)
    search = tmp_path / 'search/BP'
    search.mkdir(parents=True)
    (search / 'protocol.json').write_text(json.dumps(metadata))
    results = search / 'search_results.json'
    results.write_text(json.dumps([{'score': .2, 'hparams': {'identity_a': .2}},
                                   {'score': .8, 'hparams': {'identity_a': .7}}]))
    return results, metadata, current, old, new


def test_reuse_requires_explicit_opt_in_and_preserves_historical_evidence(handoff):
    results, saved, current, old, _ = handoff
    # A failed historical preparation need not erase the search's immutable
    # sidecar: use its pinned hash and compare against independently valid new evidence.
    report_path = Path(old['prepared_dir']) / 'pdb_train_evidence_report.json'
    report_path.write_text('{"status": "failed"}')
    snapshots = {p: p.read_bytes() for p in (results, results.parent / 'protocol.json',
                                            Path(old['pdb_train']['identity']), report_path)}
    with pytest.raises(ValueError, match='different cohort'):
        prepare_final_candidates(results, 1, current)
    candidates, report = prepare_final_candidates(results, 1, current, allow_relocated=True)
    assert candidates == [{'identity_a': .7}]
    assert report['comparison']['mode'] == 'verified_resource_relocation'
    assert 'pdb_train/identity' in report['comparison']['equivalent_identity_sidecars']
    assert all(p.read_bytes() == contents for p, contents in snapshots.items())
    exact, audit = prepare_final_candidates(results, 1, saved)
    assert exact == candidates and audit['comparison']['mode'] == 'exact_protocol_match'


@pytest.mark.parametrize('change', ['cohort', 'exclusions', 'vocab', 'settings', 'reference', 'input'])
def test_reuse_rejects_changed_protocol_or_inputs(handoff, change):
    results, _, current, _, _ = handoff
    if change == 'cohort':
        current['train_ids_sha256'] = 'another cohort'
    elif change == 'exclusions':
        current['validation_exclude_ids'] = []
    elif change == 'vocab':
        current['go_terms'] = ['g2']
    elif change == 'settings':
        current['configuration']['final_epochs'] = 2
    elif change == 'reference':
        current['reference_source_hashes']['predict.py']['sha256'] = 'changed'
    else:
        current['resources']['pdb_train/fasta']['sha256'] = 'changed'
    with pytest.raises(ValueError):
        prepare_final_candidates(results, 1, current, allow_relocated=True)


def test_reuse_rejects_tampered_historical_sidecar(handoff):
    results, _, current, old, _ = handoff
    path = Path(old['pdb_train']['identity'])
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='Historical identity sidecar differs'):
        prepare_final_candidates(results, 1, current, allow_relocated=True)


def test_reuse_rejects_valid_but_different_identity_evidence(handoff, tmp_path):
    results, _, current, _, new = handoff
    other = tmp_path / 'other.tsv'
    other.write_text('q\ts\t1e-9\t100\t50\t100\t100\t50\t50\t20\n')
    new['prepared_dir'] = str(tmp_path / 'changed')
    new['pdb_train']['enriched_hits'] = str(other)
    new['pdb_train']['identity'] = str(tmp_path / 'changed/identity.json')
    prepare_identity_split(resources=new, split='pdb_train')
    current['resources']['pdb_train/identity'] = resource_fingerprint(new['pdb_train']['identity'])
    with pytest.raises(ValueError, match='Identity records, retained-hit evidence, or policy changed'):
        prepare_final_candidates(results, 1, current, allow_relocated=True)


def test_reuse_still_requires_valid_current_report(handoff):
    results, _, current, _, new = handoff
    (Path(new['prepared_dir']) / 'pdb_train_evidence_report.json').write_text('{}')
    with pytest.raises(ValueError, match='evidence report checksum mismatch'):
        prepare_final_candidates(results, 1, current, allow_relocated=True)


def test_search_directory_override_and_cli_forwarding(monkeypatch, tmp_path):
    assert search_results_path({'base_dir_search': 'default'}, 'BP', 'existing') == Path('existing/BP/search_results.json')
    monkeypatch.chdir(fusion_baselines.ROOT)
    monkeypatch.setattr(fusion_baselines, 'require_training_inputs', lambda *a, **kw: None)
    calls = []
    monkeypatch.setattr(fusion_baselines.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    fusion_baselines.main(['final', '--model', 'identity', '--search-dir', 'existing', '--check-only'])
    assert len(calls) == 3
    for cmd in calls:
        assert cmd[cmd.index('--search-dir') + 1] == 'existing'
        assert '--check-only' in cmd
        assert cmd[cmd.index('--run_type') + 1] == 'full_training'


def test_failed_repreparation_preserves_report_and_sidecar(handoff, tmp_path):
    _, _, _, old, _ = handoff
    report = Path(old['prepared_dir']) / 'pdb_train_evidence_report.json'
    identity = Path(old['pdb_train']['identity'])
    before = report.read_bytes(), identity.read_bytes()
    # Moving an identical source changes provenance, but must not overwrite the
    # report attached to the existing sidecar on an unsuccessful preparation.
    moved = tmp_path / 'moved.tsv'
    moved.write_bytes(Path(old['pdb_train']['enriched_hits']).read_bytes())
    old['pdb_train']['enriched_hits'] = str(moved)
    with pytest.raises(ValueError, match='Stale sidecar'):
        prepare_identity_split(resources=old, split='pdb_train')
    assert (report.read_bytes(), identity.read_bytes()) == before
    assert (Path(old['prepared_dir']) / 'pdb_train_preparation_failure.json').is_file()
    from models.identity_fusion_data import load_identity_sidecar
    assert set(load_identity_sidecar(identity)) == {'q'}


@pytest.mark.parametrize('change', ['summary', 'tensor', 'file_set'])
def test_reuse_allows_only_homology_summary_changes(handoff, change):
    results, saved, current, _, _ = handoff
    saved['resources']['pdb_train/homology_shards'] = {
        'path': 'old/shards', 'files': {'homology_shard_0000.pt': 'tensor hash',
                                       'homology_shard_metadata.json': 'old summary'}}
    current['resources']['pdb_train/homology_shards'] = copy.deepcopy(saved['resources']['pdb_train/homology_shards'])
    files = current['resources']['pdb_train/homology_shards']['files']
    files['homology_shard_metadata.json'] = 'new summary'
    if change == 'tensor':
        files['homology_shard_0000.pt'] = 'changed tensor'
    elif change == 'file_set':
        files['homology_shard_0001.pt'] = 'extra tensor'
    (results.parent / 'protocol.json').write_text(json.dumps(saved))
    if change == 'summary':
        _, report = prepare_final_candidates(results, 1, current, allow_relocated=True)
        assert report['comparison']['equivalent_homology_shards']['pdb_train/homology_shards']['tensor_file_count'] == 1
    else:
        with pytest.raises(ValueError, match='Homology shard'):
            prepare_final_candidates(results, 1, current, allow_relocated=True)
