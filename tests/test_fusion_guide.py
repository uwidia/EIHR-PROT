"""Keep the published preparation commands aligned with their configuration."""
from pathlib import Path
import shlex

import yaml

from scripts.fusion_baselines import ASPECTS, SPLITS, check_resources, inference_args

ROOT = Path(__file__).resolve().parents[1]


def test_documented_enrichment_and_output_paths_match_configuration(monkeypatch):
    monkeypatch.chdir(ROOT)
    resources = yaml.safe_load(Path('configs/fusion_baseline_resources.yaml').read_text())
    guide = Path('baseline_reference.md').read_text()
    commands = [shlex.split(line) for line in guide.splitlines()
                if line.startswith('uv run --extra cu128 python scripts/fusion_baselines.py enrich ')]
    assert len(commands) == len(SPLITS)
    for command in commands:
        split = command[command.index('--splits') + 1]
        assert command[command.index('--output') + 1] == resources[split]['enriched_hits']
    for split in SPLITS:
        assert Path(resources[split]['identity']).parent == Path(resources['prepared_dir'])
    generation = Path(resources['prepared_dir']).name
    assert Path(resources['comparison_dir']).name == generation
    assert Path(resources['evaluation_dir']).name == f'fusion_{generation}'
    for model in ('fixed', 'identity'):
        config = yaml.safe_load(Path(f'configs/sequence_homology_{model}_fusion.yaml').read_text())
        assert config['resources'] == 'configs/fusion_baseline_resources.yaml'
        for key in ('base_dir_search', 'base_dir_final'):
            assert Path(config[key]).parent.name == generation
        assert config['base_dir_final'] + '/{aspect}/best_model.pt' in guide
        for dataset in ('pdb_test', 'af_test'):
            args = inference_args(resources, model, dataset, 'BP')
            assert args[args.index('--checkpoint') + 1] == config['base_dir_final'] + '/BP/best_model.pt'


def test_check_reports_missing_prerequisites_instead_of_crashing(tmp_path):
    def path(name):
        return str(tmp_path / name)
    resources = {'reference_db': path('db'), 'obo': path('go.obo'),
                 'train_annotations': path('annotations.tsv'), 'reference_source_dir': path('reference'),
                 'go_vocab': path('{aspect}/vocab.json'), 'subject_go_index': path('{aspect}/index.json'),
                 'prepared_dir': path('prepared'), 'validation_exclude_ids': ['2VAU-A', '5LSQ-A']}
    for split in SPLITS:
        resources[split] = {field: path(f'{split}/{field}') for field in
                            ('fasta', 'manifest', 'original_hits', 'enriched_hits', 'identity', 'esm_shards')}
        resources[split]['homology_shards'] = path(split + '/{aspect}/homology')
    report = check_resources(resources)
    assert not report['ready']
    for required in ('db.dmnd', 'go.obo', 'annotations.tsv', 'reference/predict.py', 'reference/alignment_knn.py'):
        assert path(required) in report['missing']
    for split in SPLITS:
        for field in ('fasta', 'manifest', 'original_hits', 'enriched_hits', 'identity'):
            assert resources[split][field] in report['missing']
    for aspect in ASPECTS:
        assert path(f'{aspect}/vocab.json') in report['missing']
    assert (tmp_path / 'prepared/resource_check.json').is_file()


def test_training_preflight_reports_wrong_reference_directory(tmp_path):
    import pytest
    from scripts.fusion_baselines import require_training_inputs
    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'predict.py').write_text('# reference\n')
    with pytest.raises(FileNotFoundError, match='reference_source_dir.*alignment_knn.py'):
        require_training_inputs({'reference_source_dir': str(reference)}, {}, ['BP'], final=False)


def test_final_preflight_requires_all_requested_searches(tmp_path):
    import pytest
    from scripts.fusion_baselines import require_training_inputs
    reference = tmp_path / 'reference'
    reference.mkdir()
    for name in ('predict.py', 'alignment_knn.py'):
        (reference / name).write_text('# reference\n')
    resources = {'reference_source_dir': str(reference)}
    config = {'base_dir_search': str(tmp_path / 'search')}
    require_training_inputs(resources, config, ['BP', 'MF'], final=False)
    bp = tmp_path / 'search/BP'
    bp.mkdir(parents=True)
    (bp / 'search_results.json').write_text('[]')
    (bp / 'protocol.json').write_text('{}')
    require_training_inputs(resources, config, ['BP'], final=True)
    with pytest.raises(FileNotFoundError, match='changing v1/v2 paths does not migrate results') as error:
        require_training_inputs(resources, config, ['BP', 'MF'], final=True)
    assert str(tmp_path / 'search/MF/search_results.json') in str(error.value)
    config['search_results_path'] = str(bp / 'search_results.json')
    require_training_inputs(resources, config, ['MF'], final=True)
