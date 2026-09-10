"""Original priors and exact-count fusion must not share identity requirements."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from models.diamond_score_baseline import compute_diamond_score_probabilities
from reliability_aware.utils.diamond_homology import (
    DIAMOND_OUTFMT_FIELDS, LEGACY_DIAMOND_OUTFMT_FIELDS, DiamondSearchConfig,
    _parse_diamond_hits, build_aligned_homology_shards, run_diamond_blastp,
)
from reliability_aware.utils.fusion_preparation import prepare_identity_split, verify_frozen_priors
from scripts.fusion_baselines import build_missing_homology


@pytest.fixture
def resources(tmp_path):
    labels = ['good', '2VAU-A', 'nohit', '5LSQ-A']
    fasta = tmp_path / 'val.fasta'
    fasta.write_text(''.join(f'>{q}\n' + 'A' * 331 + '\n' for q in labels))
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['shard_number', 'local_seq_idx', 'global_seq_idx', 'label', 'sequence_length', 'truncated_length'])
        writer.writerows([0, i, i, q, 331, 331] for i, q in enumerate(labels))
    original = tmp_path / 'original.tsv'
    original.write_text(
        'good\t1XX8-A\t1e-9\t100\t50\t331\t66\t50\t50\t25\n'
        '2VAU-A\t1XX8-A\t1.25e-33\t125\t98.2\t331\t66\t354\t28.0\t99\n'
        '5LSQ-A\t1XX8-A\t1e-9\t100\t80\t331\t66\t99\t100\t71\n')
    enriched = tmp_path / 'enriched.tsv'
    enriched.write_bytes(original.read_bytes())
    vocab, index = tmp_path / 'vocab.json', tmp_path / 'index.json'
    vocab.write_text('["GO:1"]')
    index.write_text('{"1XX8-A": [0]}')
    esm = tmp_path / 'esm'
    esm.mkdir()
    (esm / 'part_0000.pt').touch()
    return {'prepared_dir': str(tmp_path / 'prepared'), 'validation_exclude_ids': ['2VAU-A', '5LSQ-A'],
            'go_vocab': str(vocab), 'subject_go_index': str(index),
            'pdb_val': {'fasta': str(fasta), 'manifest': str(manifest), 'esm_shards': str(esm),
                        'original_hits': str(original), 'enriched_hits': str(enriched),
                        'homology_shards': str(tmp_path / 'homology'),
                        'identity': str(tmp_path / 'prepared/identity.json')}}


def build_original(resources, output, hits):
    build_aligned_homology_shards(
        resources['pdb_val']['manifest'], hits, resources['subject_go_index'],
        resources['go_vocab'], output, DiamondSearchConfig(), use_fp16=True)
    return torch.load(Path(output) / 'homology_shard_0000.pt', weights_only=False)


def test_original_shards_preserve_full_cohort_and_nine_column_values(resources, tmp_path):
    item = resources['pdb_val']
    enriched_original = Path(item['original_hits'])
    before = enriched_original.read_bytes()
    actual = build_original(resources, item['homology_shards'], enriched_original)
    legacy = tmp_path / 'legacy.tsv'
    legacy.write_text(''.join('\t'.join(row.split('\t')[:9]) + '\n' for row in enriched_original.read_text().splitlines()))
    expected = build_original(resources, tmp_path / 'legacy_shards', legacy)
    assert actual['labels'] == ['good', '2VAU-A', 'nohit', '5LSQ-A']
    for key in ('priors', 'gate_features'):
        assert all(torch.equal(a, b) for a, b in zip(actual[key], expected[key]))
    assert actual['debug_hits'] == expected['debug_hits']
    assert [s['n_hits'] for s in actual['stats']] == [1, 1, 0, 1]
    assert enriched_original.read_bytes() == before
    prepare_identity_split(resources=resources, split='pdb_val')
    assert verify_frozen_priors(resources, 'pdb_val', 'BP')['checked_n'] == 2
    scores = compute_diamond_score_probabilities(query_labels=actual['labels'], go_terms=['GO:1'],
        diamond_hits_path=enriched_original, subject_go_index_path=resources['subject_go_index'])
    assert scores[:, 0].tolist() == [1., 1., 0., 1.]


def test_fusion_local_shards_exclude_evidence_without_reindexing(resources):
    item = resources['pdb_val']
    before = Path(item['enriched_hits']).read_bytes()
    report = prepare_identity_split(resources=resources, split='pdb_val')
    assert report['excluded_query_ids'] == ['2VAU-A', '5LSQ-A']
    build_missing_homology(resources, ['pdb_val'], ['BP'])
    shard = torch.load(Path(item['homology_shards']) / 'homology_shard_0000.pt', weights_only=False)
    assert shard['labels'] == ['good', '2VAU-A', 'nohit', '5LSQ-A']
    assert [s['n_hits'] for s in shard['stats']] == [1, 0, 0, 0]
    records = json.loads(Path(item['identity']).read_text())['records']
    assert [r['protein_id'] for r in records] == ['good', 'nohit']
    assert Path(item['enriched_hits']).read_bytes() == before


def test_fusion_rejects_stale_unfiltered_working_copy(resources):
    prepare_identity_split(resources=resources, split='pdb_val')
    working = Path(resources['prepared_dir']) / 'pdb_val_original_hits.tsv'
    working.write_bytes(Path(resources['pdb_val']['original_hits']).read_bytes())
    with pytest.raises(ValueError, match='Existing prepared artifact differs'):
        build_missing_homology(resources, ['pdb_val'], ['BP'])


@pytest.mark.parametrize('count', ['99', '99.0', 'unavailable'])
def test_original_ignores_identity_column_but_identity_stays_strict(tmp_path, count):
    hits = tmp_path / 'hits.tsv'
    hits.write_text(f'2VAU-A\t1XX8-A\t1e-9\t100\t98.2\t331\t66\t354\t28\t{count}\n')
    assert _parse_diamond_hits(hits, read_nident=False)['2VAU-A'][0].nident is None
    with pytest.raises(ValueError):
        _parse_diamond_hits(hits)


@pytest.mark.parametrize('enriched', [False, True])
def test_search_schema_is_explicit(tmp_path, enriched):
    with patch('reliability_aware.utils.diamond_homology._run_subprocess') as run:
        kwargs = {'include_nident': True} if enriched else {}
        run_diamond_blastp(tmp_path / 'query.fasta', tmp_path / 'db', tmp_path / 'hits.tsv',
                          DiamondSearchConfig(), **kwargs)
    command = run.call_args.args[0]
    fields = command[command.index('--outfmt') + 2:command.index('--evalue')]
    assert fields == (DIAMOND_OUTFMT_FIELDS if enriched else LEGACY_DIAMOND_OUTFMT_FIELDS)


def test_enrichment_entry_point_keeps_exact_count_schema(tmp_path):
    from scripts.fusion_baselines import enrich
    query = tmp_path / 'val.fasta'
    query.write_text('>q\nAAA\n')
    db = tmp_path / 'train_db'
    db.with_suffix('.dmnd').write_bytes(b'test database')
    output = tmp_path / 'nident' / 'val_hits.tsv'
    resources = {'reference_db': str(db), 'pdb_val': {'fasta': str(query)}}
    with patch('scripts.fusion_baselines.subprocess.check_output', return_value='DIAMOND test'), \
         patch('scripts.fusion_baselines.subprocess.run', side_effect=lambda *a, **kw: output.write_text('q\ts\t1e-9\t100\t100\t3\t3\t3\t100\t3\n')) as run:
        enrich(resources, 'pdb_val', output, './diamond', 2)
    command = run.call_args.args[0]
    assert command[command.index('--outfmt') + 2:command.index('--evalue')] == DIAMOND_OUTFMT_FIELDS
    assert db.with_suffix('.dmnd').read_bytes() == b'test database'
    assert json.loads(output.with_suffix('.provenance.json').read_text())['fields'] == DIAMOND_OUTFMT_FIELDS
