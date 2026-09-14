import json
from pathlib import Path

import torch
import pytest

from reliability_aware.utils.fusion_protocol import VALIDATION_EXCLUSIONS, load_resources
from scripts.migrate_fusion_metadata import migrate, digest, tensor_payloads, cohort_hash


def test_default_resources_keep_internal_policy():
    resources = load_resources('configs/fusion_baseline_resources.yaml')
    assert set(resources['validation_exclude_ids']) == VALIDATION_EXCLUSIONS


@pytest.mark.parametrize('interrupted', [False, True])
def test_metadata_migration_preserves_tensors_and_updates_dependencies(tmp_path, monkeypatch, interrupted):
    root = tmp_path / 'runs'
    folder = root / 'sequence_homology_identity_fusion' / 'validation_exclude_two_v1'
    folder.mkdir(parents=True)
    evidence = folder / 'evidence.json'
    evidence.write_text(json.dumps({'excluded_query_ids': sorted(VALIDATION_EXCLUSIONS),
        'remaining_query_count': 2, 'original_query_count': 4,
        'removed_original_rows': [{'query_id': q} for q in VALIDATION_EXCLUSIONS]}))
    checkpoint = folder / 'model.pt'
    torch.save({'model_state_dict': {'weight': torch.arange(4)}, 'score': .75,
        'validation_exclude_ids': sorted(VALIDATION_EXCLUSIONS),
        'fusion_protocol_version': 'validation_exclude_two_v1',
        'resources': {'evidence': {'path': str(evidence), 'sha256': digest(evidence)}}}, checkpoint)
    payloads = tensor_payloads(checkpoint)
    protocol = folder / 'protocol.json'
    protocol.write_text(json.dumps({'checkpoint_sha256': digest(checkpoint)}))
    if interrupted:
        import os
        replace = os.replace
        def fail_on_protocol(source, target):
            if Path(target).name == 'protocol.json':
                raise PermissionError('simulated interruption')
            return replace(source, target)
        with monkeypatch.context() as context:
            context.setattr(os, 'replace', fail_on_protocol)
            with pytest.raises(PermissionError):
                migrate(root)
        assert (root / '.fusion_metadata_migration.json').is_file()
    migrate(root)
    folder = folder.with_name('cohort_v1')
    checkpoint = folder / 'model.pt'
    assert tensor_payloads(checkpoint) == payloads
    saved = torch.load(checkpoint, weights_only=False)
    assert saved['score'] == .75
    assert saved['validation_policy_sha256'] == cohort_hash(VALIDATION_EXCLUSIONS)
    assert saved['resources']['evidence']['sha256'] == digest(folder / 'evidence.json')
    assert Path(saved['resources']['evidence']['path']) == folder / 'evidence.json'
    assert json.loads((folder / 'protocol.json').read_text())['checkpoint_sha256'] == digest(checkpoint)
    assert not any(q in (folder / 'evidence.json').read_text() for q in VALIDATION_EXCLUSIONS)
    before = {p: p.read_bytes() for p in folder.iterdir()}
    migrate(root)
    assert all(p.read_bytes() == value for p, value in before.items())


def test_resource_check_does_not_display_internal_policy(tmp_path, capsys):
    from scripts.fusion_baselines import check_resources, SPLITS
    fasta = tmp_path / 'queries.fasta'
    fasta.write_text(''.join(f'>{q}\nAAA\n' for q in VALIDATION_EXCLUSIONS))
    annotations = tmp_path / 'annotations.tsv'
    annotations.write_text('other\tGO:1\n')
    missing = str(tmp_path / 'missing')
    resources = {key: missing for key in ('reference_db', 'obo', 'reference_source_dir', 'go_vocab', 'subject_go_index')}
    resources.update(train_annotations=str(annotations), prepared_dir=str(tmp_path / 'prepared'),
                     validation_exclude_ids=sorted(VALIDATION_EXCLUSIONS))
    for split in SPLITS:
        resources[split] = {key: missing for key in ('fasta', 'manifest', 'esm_shards', 'original_hits', 'enriched_hits', 'identity', 'homology_shards')}
    resources['pdb_val']['fasta'] = str(fasta)
    report = check_resources(resources)
    output = capsys.readouterr().out
    assert not any(q in output or q in json.dumps(report) for q in VALIDATION_EXCLUSIONS)
    assert 'validation_excluded' not in output
