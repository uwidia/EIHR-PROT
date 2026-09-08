#!/usr/bin/env python3
"""Portable, isolated entry point for the fixed and identity fusion study."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

SPLITS = ('pdb_train', 'pdb_val', 'pdb_test', 'af_test')
ASPECTS = ('BP', 'MF', 'CC')


def configured_model(model):
    model_id = f'sequence_homology_{model}_fusion'
    path = Path('configs') / f'{model_id}.yaml'
    return model_id, path, yaml.safe_load(path.read_text())


def manifest_coverage(item):
    from reliability_aware.utils.diamond_homology import read_fasta_as_dict
    ids = set(read_fasta_as_dict(item['fasta']))
    with Path(item['manifest']).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    labels = [r['label'] for r in rows]
    if len(labels) != len(set(labels)) or set(labels) != ids:
        raise ValueError(f"Manifest does not cover the complete configured FASTA: {item['manifest']}")
    shards = {int(r['shard_number']) for r in rows}
    for shard in shards:
        path = Path(item['esm_shards']) / f'part_{shard:04d}.pt'
        if not path.is_file():
            raise FileNotFoundError(path)
    return len(ids)


def check_resources(resources):
    from reliability_aware.utils.diamond_homology import read_fasta_as_dict
    report = {'splits': {}, 'missing': []}
    for split in SPLITS:
        item = resources[split]
        ids = set(read_fasta_as_dict(item['fasta']))
        report['splits'][split] = {'fasta_n': len(ids), 'validation_excluded': sorted(ids & set(resources['validation_exclude_ids'])) if split == 'pdb_val' else []}
        try:
            report['splits'][split]['manifest_n'] = manifest_coverage(item)
        except (ValueError, FileNotFoundError) as exc:
            report['missing'].append(str(exc))
        for aspect in ASPECTS:
            path = Path(item['homology_shards'].format(aspect=aspect))
            if not any(path.glob('homology_shard_*.pt')):
                report['missing'].append(str(path))
        annotation = item.get('annotations', resources['train_annotations'])
        # Annotation TSV has a comment preamble; the first field still identifies rows.
        annotated = {line.split('\t')[0] for line in Path(annotation).read_text().splitlines() if '\t' in line and not line.startswith('#')}
        report['splits'][split]['annotation_ids_present'] = len(ids & annotated)
        report['splits'][split]['annotation_ids_absent'] = sorted(ids - annotated)
    root = Path(resources['prepared_dir'])
    root.mkdir(parents=True, exist_ok=True)
    (root / 'resource_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def enrich(resources, split, output, executable, threads):
    from reliability_aware.utils.diamond_homology import DIAMOND_OUTFMT_FIELDS
    from reliability_aware.utils.prediction_cache import sha256_file
    if output is None:
        raise ValueError('enrich requires --output pointing to a NEW file; then set enriched_hits in the resources YAML')
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    db = Path(resources['reference_db'])
    query = Path(resources[split]['fasta'])
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, 'blastp', '--db', str(db), '--query', str(query), '--out', str(output),
               '--outfmt', '6', *DIAMOND_OUTFMT_FIELDS, '--evalue', '1e-5', '--max-target-seqs', '50',
               '--sensitive', '--iterate', '--threads', str(threads)]
    version = subprocess.check_output([executable, 'version'], text=True).strip()
    metadata = {'command': command, 'version': version, 'fields': DIAMOND_OUTFMT_FIELDS,
                'schema': 'eihr-diamond-hits-v2-nident', 'query_sha256': sha256_file(query),
                'database_sha256': sha256_file(db.with_suffix('.dmnd')), 'status': 'running'}
    meta = output.with_suffix('.provenance.json')
    meta.write_text(json.dumps(metadata, indent=2))
    subprocess.run(command, check=True)
    metadata.update({'output_sha256': sha256_file(output), 'status': 'search_complete; retained evidence validation still required'})
    meta.write_text(json.dumps(metadata, indent=2))
    print(f'Set {split}.enriched_hits to {output}, then run prepare. Existing priors and DB were not changed.')


def build_missing_homology(resources, splits, aspects):
    from reliability_aware.utils.diamond_homology import DiamondSearchConfig, build_aligned_homology_shards
    for split in splits:
        item = resources[split]
        manifest_coverage(item)
        for aspect in aspects:
            output = Path(item['homology_shards'].format(aspect=aspect)).resolve()
            if output.is_relative_to((ROOT / 'diamond_db').resolve()):
                raise ValueError('Missing-resource construction requires a new runs/fusion_resources/... destination in the YAML; original diamond_db is read-only')
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f'Refusing to replace existing homology resources: {output}')
            # The validation copies omit whole excluded queries; original manifest indices remain intact.
            hits = Path(resources['prepared_dir']) / f'{split}_original_hits.tsv'
            if not hits.exists():
                raise FileNotFoundError(f'Run prepare for {split} first: {hits}')
            build_aligned_homology_shards(manifest_path=item['manifest'], diamond_hits=hits,
                subject_go_index_json_path=resources['subject_go_index'].format(aspect=aspect),
                go_vocab_json_path=resources['go_vocab'].format(aspect=aspect), output_dir=output,
                config=DiamondSearchConfig(), exclude_self_hits=split == 'pdb_train',
                use_fp16=True, keep_debug_hits=True)


def inference_args(resources, model, dataset, aspect):
    model_id, _, config = configured_model(model)
    item = resources[dataset]
    outdir = Path(resources['evaluation_dir']) / model_id / dataset / aspect
    checkpoint = Path(config['base_dir_final']) / aspect / 'best_model.pt'
    args = ['--ablation', model_id, '--go_aspect', aspect, '--mode', 'evaluate', '--no-bootstrap',
            '--dataset_id', dataset, '--checkpoint', str(checkpoint), '--test_fasta', item['fasta'],
            '--train_fasta', resources['pdb_train']['fasta'], '--test_manifest_path', item['manifest'],
            '--test_esm_shard_dir', item['esm_shards'], '--test_homology_shard_dir', item['homology_shards'].format(aspect=aspect),
            '--go_vocab_path', resources['go_vocab'].format(aspect=aspect), '--go_annotation_path', item['annotations'],
            '--train_go_annotation_path', resources['train_annotations'], '--obo_path', resources['obo'],
            '--outdir', str(outdir), '--prediction_cache_path', str(outdir / 'predictions.npz')]
    if model == 'identity':
        args += ['--identity_sidecar_path', item['identity']]
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'prepare', 'enrich', 'verify-priors', 'build-missing-homology', 'search', 'final', 'infer', 'reference', 'compare'])
    parser.add_argument('--resources', type=Path, default=Path('configs/fusion_baseline_resources.yaml'))
    parser.add_argument('--model', choices=['fixed', 'identity'])
    parser.add_argument('--dataset', choices=['pdb_test', 'af_test'])
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=list(SPLITS))
    parser.add_argument('--aspects', nargs='+', choices=ASPECTS, default=list(ASPECTS))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--diamond', default='./diamond')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--n-resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        parser.error(f'Run from the repository root: {ROOT}')
    from reliability_aware.utils.fusion_protocol import load_resources
    resources = load_resources(args.resources)
    if args.action == 'check':
        check_resources(resources)
    elif args.action == 'prepare':
        from reliability_aware.utils.fusion_preparation import prepare_identity_split
        from reliability_aware.utils.prediction_cache import sha256_file
        failures = []
        for split in args.splits:
            try:
                report = prepare_identity_split(resources=resources, split=split)
                print(f"{split}: {report['status']}; {report['remaining_query_count']} proteins")
            except (ValueError, FileNotFoundError) as exc:
                failures.append(f'{split}: {exc}')
        root = Path(resources['reference_source_dir'])
        hashes = {name: sha256_file(root / name) for name in ('predict.py', 'alignment_knn.py') if (root / name).is_file()}
        Path(resources['prepared_dir'], 'interlabelgo_source.json').write_text(json.dumps(hashes, indent=2))
        if failures:
            raise ValueError('\n'.join(failures))
    elif args.action == 'enrich':
        if len(args.splits) != 1:
            parser.error('enrich requires exactly one --splits value')
        enrich(resources, args.splits[0], args.output, args.diamond, args.threads)
    elif args.action == 'verify-priors':
        from reliability_aware.utils.fusion_preparation import verify_frozen_priors
        for split in args.splits:
            for aspect in args.aspects:
                report = verify_frozen_priors(resources, split, aspect)
                print(f"{split}/{aspect}: {report['checked_n']} unchanged priors verified")
    elif args.action == 'build-missing-homology':
        build_missing_homology(resources, args.splits, args.aspects)
    elif args.action in ('search', 'final'):
        if args.model is None:
            parser.error('search/final require --model')
        model_id, path, config = configured_model(args.model)
        if Path(config['resources']).resolve() != args.resources.resolve():
            raise ValueError('The model YAML resources path must match --resources')
        print(f"Budget: {len(args.aspects)} aspects; {config['num_trials']} search trials x {config['trial_epochs']} epochs; up to {config['top_k_params']} final candidates x {config['final_epochs']} epochs per aspect", flush=True)
        for aspect in args.aspects:
            subprocess.run([sys.executable, 'scripts/run_model_training.py', '--ablation', model_id,
                '--go_aspect', aspect, '--hparams', str(path), '--run_type',
                'randomized_search' if args.action == 'search' else 'full_training'], check=True)
    elif args.action == 'reference':
        if args.dataset is None:
            parser.error('reference requires --dataset')
        manifest_coverage(resources[args.dataset])
        for aspect in args.aspects:
            checkpoint = resources['reference_checkpoints'][aspect]
            cache = resources['reference_caches'][args.dataset][aspect]
            if not checkpoint or not cache:
                raise ValueError('Configure reference_checkpoints and reference_caches in the resources YAML')
            if Path(cache).exists():
                raise FileExistsError(f'{cache}: use compare to reuse completed caches')
            argv = inference_args(resources, 'fixed', args.dataset, aspect)
            for flag, value in [('--ablation', 'sequence_homology_confidence_gate'), ('--checkpoint', checkpoint),
                                ('--outdir', str(Path(cache).parent)), ('--prediction_cache_path', cache)]:
                argv[argv.index(flag) + 1] = str(value)
            subprocess.run([sys.executable, 'scripts/inference/run_inference_seq_hom.py', *argv], check=True)
    elif args.action == 'infer':
        if args.model is None or args.dataset is None:
            parser.error('infer requires --model and --dataset')
        manifest_coverage(resources[args.dataset])
        for aspect in args.aspects:
            from reliability_aware.utils.fusion_preparation import verify_frozen_priors
            verify_frozen_priors(resources, args.dataset, aspect)
            argv = inference_args(resources, args.model, args.dataset, aspect)
            cache = Path(argv[argv.index('--prediction_cache_path') + 1])
            if cache.exists() or cache.with_suffix('.json').exists():
                raise FileExistsError(f'{cache}: use compare for cache-only reruns; new inputs require a new evaluation_dir')
            subprocess.run([sys.executable, 'scripts/inference/run_inference_seq_hom.py', *argv], check=True)
    else:
        from reliability_aware.utils.fusion_reporting import compare_all
        print(compare_all(resources, n_resamples=args.n_resamples, seed=args.seed))


if __name__ == '__main__':
    main()
