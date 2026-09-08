"""Whole-test fusion comparisons using the existing paired CAFA bootstrap engine."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from openpyxl import Workbook

from reliability_aware.utils.bootstrap import PreparedCafaEvaluator, ResamplingPlan, percentile_interval
from reliability_aware.utils.bootstrap_reporting import _sheet
from reliability_aware.utils.prediction_cache import align_caches, read_prediction_cache, sha256_file

METRICS = ('Fmax', 'AUPR', 'Smin')


def compare_pair(full, baseline, *, dataset, aspect, obo, output_dir, n_resamples=10000, seed=42):
    full, baseline = align_caches(full, baseline)
    for cache in (full, baseline):
        if cache.metadata.get('dataset') not in ({'pdb_test', 'PDB'} if dataset == 'pdb_test' else {'af_test', 'AF'}):
            raise ValueError('Cache dataset mismatch')
        if cache.metadata.get('go_aspect') != aspect:
            raise ValueError('Cache GO aspect mismatch')
        if cache.metadata.get('source_hashes', {}).get('obo', {}).get('sha256') != sha256_file(Path(obo)):
            raise ValueError('Cache OBO provenance mismatch')
    for name in ('test_fasta', 'annotations'):
        f = full.metadata.get('source_hashes', {}).get(name, {}).get('sha256')
        b = baseline.metadata.get('source_hashes', {}).get(name, {}).get('sha256')
        if not f or f != b:
            raise ValueError(f'Cache {name} provenance mismatch')
    canonical_train = lambda c: sorted(tuple(sorted(x)) for x in c.metadata['train_annotations'])
    if canonical_train(full) != canonical_train(baseline):
        raise ValueError('Caches use different fixed training annotations/IC')
    prepared = []
    eligible = []
    for cache in (full, baseline):
        p, mask = PreparedCafaEvaluator.prepare(y_true=cache.labels, y_prob=cache.probabilities,
                go_terms=cache.go_terms.tolist(), go_aspect=aspect, obo_path=Path(obo),
                train_annotations=cache.metadata['train_annotations'])
        if not np.array_equal(mask, cache.eligibility):
            raise ValueError('Cache eligibility disagrees with CAFA evaluator')
        prepared.append(p)
        eligible.append(mask)
    n = int(eligible[0].sum())
    if not n:
        return [{'metric': m, 'n': 0, 'status': 'NA: no eligible proteins'} for m in METRICS], {}
    plan = ResamplingPlan.load_or_create(Path(output_dir) / f'{dataset}_{aspect}_seed{seed}_B{n_resamples}.npz',
            dataset=dataset, aspect=aspect, cohort_name='complete_test',
            protein_ids=full.protein_ids[eligible[0]], n_resamples=n_resamples, seed=seed)
    points = [p.metrics(np.ones(n, dtype=np.int32)) for p in prepared]
    reps = [p.bootstrap(plan) for p in prepared]
    rows, arrays = [], {}
    for metric in METRICS:
        delta = reps[0][metric] - reps[1][metric]
        lo, hi = percentile_interval(delta)
        rows.append({'metric': metric, 'n': n, 'full': points[0][metric], 'baseline': points[1][metric],
                     'delta': points[0][metric] - points[1][metric], 'ci_lower': lo, 'ci_upper': hi,
                     'ci': f'[{lo:.6f}, {hi:.6f}]', 'status': 'ok'})
        arrays[metric + '_full'] = reps[0][metric]
        arrays[metric + '_baseline'] = reps[1][metric]
        arrays[metric + '_delta'] = delta
    return rows, arrays


def write_fusion_workbook(rows, parameters, provenance, path):
    if len(rows) != 36:
        raise ValueError('Fusion comparison workbook requires all 36 metric/status rows')
    book = Workbook()
    book.remove(book.active)
    keys = ['dataset', 'aspect', 'n', 'comparator', 'metric', 'full', 'baseline', 'delta', 'ci_lower', 'ci_upper', 'ci', 'status']
    for sheet, subset in [('Performance', rows), ('PDB', [r for r in rows if r['dataset'] == 'pdb_test']),
                          ('AF', [r for r in rows if r['dataset'] == 'af_test'])]:
        _sheet(book, sheet, keys, [[r.get(k) for k in keys] for r in subset])
    for name, model, fields in [('Fixed_weights', 'fixed', ['alpha_neural', 'alpha_homology']),
                                ('Identity_parameters', 'identity', ['identity_a', 'identity_k'])]:
        _sheet(book, name, ['aspect', *fields, 'status'],
               [[a, *[parameters.get((model, a), {}).get(k) for k in fields],
                 'ok' if (model, a) in parameters else 'NA: no completed cache'] for a in ('BP', 'MF', 'CC')])
    _sheet(book, 'Provenance', ['key', 'value'], [[k, json.dumps(v, sort_keys=True)] for k, v in provenance.items()])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    return path


def compare_all(resources, *, n_resamples=10000, seed=42):
    output = Path(resources['comparison_dir'])
    output.mkdir(parents=True, exist_ok=True)
    rows, parameters, checkpoint_hashes = [], {}, {}
    provenance = {'resources': resources, 'B': n_resamples, 'seed': seed, 'CI': '95% percentile; full minus baseline; negative Smin favors full', 'inputs': {}}
    for dataset in ('pdb_test', 'af_test'):
        for aspect in ('BP', 'MF', 'CC'):
            for model in ('fixed', 'identity'):
                model_id = f'sequence_homology_{model}_fusion'
                reference = resources['reference_caches'][dataset][aspect]
                comparator = Path(resources['evaluation_dir']) / model_id / dataset / aspect / 'predictions.npz'
                key = f'{dataset}/{aspect}/{model}'
                try:
                    if reference is None:
                        raise FileNotFoundError('reference cache not configured')
                    full, baseline = read_prediction_cache(Path(reference)), read_prediction_cache(comparator)
                    if full.metadata.get('model_id') != 'sequence_homology_confidence_gate' or baseline.metadata.get('model_id') != model_id:
                        raise ValueError('Cache model ID mismatch')
                    if baseline.metadata.get('validation_exclude_ids') != ['2VAU-A', '5LSQ-A']:
                        raise ValueError('Baseline cache uses a different validation cohort')
                    pair, reps = compare_pair(full, baseline, dataset=dataset, aspect=aspect,
                            obo=resources['obo'], output_dir=output / 'plans', n_resamples=n_resamples, seed=seed)
                    digest = baseline.metadata.get('checkpoint_sha256')
                    if not digest or ((model, aspect) in checkpoint_hashes and checkpoint_hashes[(model, aspect)] != digest):
                        raise ValueError('PDB and AF must use the same selected baseline checkpoint')
                    checkpoint_hashes[(model, aspect)] = digest
                    full_digest = full.metadata.get('checkpoint_sha256')
                    if not full_digest or (('full', aspect) in checkpoint_hashes and checkpoint_hashes[('full', aspect)] != full_digest):
                        raise ValueError('PDB and AF must use the same full-model reference checkpoint')
                    checkpoint_hashes[('full', aspect)] = full_digest
                    params = baseline.metadata['fusion_parameters']
                    if (model, aspect) in parameters and parameters[(model, aspect)] != params:
                        raise ValueError('PDB and AF caches use different selected parameters')
                    parameters[(model, aspect)] = params
                    provenance['inputs'][key] = {'full': full.metadata, 'baseline': baseline.metadata,
                            'cache_sha256': sha256_file(comparator), 'reference_sha256': sha256_file(Path(reference))}
                    np.savez_compressed(output / f'{dataset}_{aspect}_{model}_replicates.npz', **reps)
                except FileNotFoundError as exc:
                    pair = [{'metric': m, 'status': f'NA: {exc}'} for m in METRICS]
                rows.extend(dict(r, dataset=dataset, aspect=aspect, comparator=model_id) for r in pair)
    (output / 'comparisons.json').write_text(json.dumps(rows, indent=2))
    return write_fusion_workbook(rows, parameters, provenance, output / 'fusion_baseline_comparisons.xlsx')
