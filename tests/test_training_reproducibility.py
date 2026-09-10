"""Real short training runs, with dropout, replay across fresh Python processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from reliability_aware.utils.reproducibility import resolve_cpu_threads, training_environment


TRAIN_SCRIPT = r'''
import sys
from reliability_aware.utils.reproducibility import prepare_training_process
seed = int(sys.argv[2])
cpu_threads = int(sys.argv[3])
prepare_training_process(seed, cpu_threads)

import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.sequence_homology_ablation import build_sequence_homology_confidence_gate_model
from reliability_aware.utils import model_training as training
from reliability_aware.utils.generic_loader import HybridBatchSampler
from reliability_aware.utils.reproducibility import configure_training, seed_training_rng
from scripts.run_confidence_gate_ablations import VARIANTS

root = Path(sys.argv[1])
configure_training(seed, cpu_threads=cpu_threads)

class TinyDataset(Dataset):
    indices_by_shard = {key: [2*int(key), 2*int(key)+1] for key in {'0', '1', '2'}}
    lengths = [3]*6
    def __len__(self):
        return 6
    def __getitem__(self, idx):
        return dict(padded=torch.arange(3*1280).reshape(3,1280).float().sin() + idx/10,
                    mask=torch.ones(3, dtype=torch.bool),
                    homology_scores=torch.tensor([.2, .8]),
                    gate_features=torch.tensor([70.+idx, .8, 1.6, 1.]),
                    targets=torch.tensor([float(idx%2), float((idx+1)%2)]), index=idx)

def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

def state_digest(model, neural_only=False):
    return {k: digest(v) for k,v in model.state_dict().items()
            if not neural_only or k.startswith(('seq_branch.', 'head.'))}

results = {}
for variant, omitted in VARIANTS.items():
    # Different pre-run RNG consumption must not affect the experiment.
    random.random(); np.random.rand(11); torch.rand(17)
    ds = TinyDataset()
    loaders = [DataLoader(ds, batch_sampler=HybridBatchSampler(ds, 2, seed=991)) for _ in range(2)]
    trace = {'dropout': [], 'batches': [], 'evaluation_rng': []}
    def builder(hp, terms, device):
        model, optimizer = build_sequence_homology_confidence_gate_model(hp, terms, device)
        trace['initial_neural'] = state_digest(model, True)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.register_forward_pre_hook(lambda mod, args: trace['dropout'].append(digest(torch.get_rng_state())))
        return model, optimizer
    original_step = training.train_one_epoch
    def step(**kwargs):
        trace['batches'].append(list(kwargs['loader'].batch_sampler))
        return original_step(**kwargs)
    training.train_one_epoch = step
    def evaluate(**kwargs):
        # These draws must also match across variants and fresh processes.
        trace['evaluation_rng'].append([random.random(), float(np.random.rand()), float(torch.rand(()))])
        return dict(Fmax=.1*len(trace['evaluation_rng']), AUPR=.3, Smin=.4, val_loss=.5)
    training.evaluate_model = evaluate
    hp = dict(learning_rate=.001, attn_hidden_dim=4, head_hidden_dim=4,
              gate_hidden_dim=4, attn_dropout=.2, head_dropout=.3, gate_dropout=.2,
              pos_weight_cap=5, lambda_hier=0., initialization_seed=seed,
              training_seed=seed, reproducible_training=True, omitted_gate_feature=omitted,
              cpu_threads=cpu_threads)
    directory = root / variant
    training.run_model_training(
        [hp], train_loader=loaders[0], val_loader=loaders[1],
        train_keep_ids_for_aspect={'a','b'}, train_label_to_indices={'a':[0], 'b':[1]},
        go_terms=['GO:1','GO:2'], child_parent_pairs=torch.empty(0,2,dtype=torch.long),
        go_aspect='BP', obo_path=None, train_annotations=[], build_model_fn=builder,
        fit_function=training.fit_model, device=torch.device('cpu'), final_epochs=2,
        patience=3, base_dir=directory, ablation='sequence_homology_confidence_gate')
    training.train_one_epoch = original_step
    checkpoint = torch.load(directory/'best_model.pt', weights_only=False)
    trace['final'] = {k: digest(v) for k,v in checkpoint['model_state_dict'].items()}
    trace['history'] = torch.load(directory/'run_000/history.pt', weights_only=False)
    trace['reproducibility'] = checkpoint['hparams']['reproducibility']
    assert checkpoint['hparams']['training_seed'] == seed
    assert checkpoint['hparams']['initialization_seed'] == seed
    assert checkpoint['hparams']['omitted_gate_feature'] == omitted
    assert json.loads((directory/'best_model_metadata.json').read_text())['hparams']['reproducibility'] == trace['reproducibility']
    results[variant] = trace

# Workers draw from all RNGs and must replay with the same generator seeds.
class RandomDataset(Dataset):
    def __len__(self): return 4
    def __getitem__(self, idx): return [random.random(), np.random.rand(), float(torch.rand(()))]
loader = DataLoader(RandomDataset(), batch_size=2, num_workers=2)
seed_training_rng(seed, loader)
first = [[v.tolist() for v in batch] for batch in loader]
seed_training_rng(seed, loader)
assert first == [[v.tolist() for v in batch] for batch in loader]
# Iterating deterministic data does not consume the global torch RNG.
loader = DataLoader(TinyDataset(), batch_size=2)
seed_training_rng(seed, loader)
before = torch.get_rng_state().clone()
list(loader)
assert torch.equal(before, torch.get_rng_state())
(root/'result.json').write_text(json.dumps(results))
'''


def test_seeded_training_replays_with_matched_dropout_batches_and_checkpoints(tmp_path):
    script = tmp_path / 'train_probe.py'
    script.write_text(TRAIN_SCRIPT)
    environment = dict(os.environ, PYTHONPATH=str(Path.cwd()))
    environment['PYTHONHASHSEED'] = 'random'  # Exercise automatic process restart.
    summaries = []
    for name, seed in [('first', 42), ('repeat', 42), ('other_seed', 43)]:
        out = tmp_path / name
        result = subprocess.run([sys.executable, str(script), str(out), str(seed), "4"],
                                env=environment, text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        summaries.append(json.loads((out/'result.json').read_text()))
    assert summaries[0] == summaries[1]
    full = summaries[0]['full']
    for variant in summaries[0].values():
        for key in ('dropout', 'batches', 'evaluation_rng', 'initial_neural'):
            assert variant[key] == full[key]
        assert variant['reproducibility']['deterministic_algorithms'] is True
        assert variant['reproducibility']['environment'] == training_environment(42, 4)
        assert variant['reproducibility']['cpu_threads'] == 4
        assert variant['reproducibility']['cpu_interop_threads'] == 4
    assert full['final'] != summaries[2]['full']['final']
    assert full['batches'] != summaries[2]['full']['batches']


@pytest.mark.parametrize('seed', [None, 'random', -1, True, 2**32])
def test_training_requires_a_resolved_valid_seed(seed):
    with pytest.raises(ValueError):
        training_environment(seed)


@pytest.mark.parametrize("available, expected", [(1, 1), (4, 4), (16, 8)])
def test_default_threads_use_available_cpu_affinity(monkeypatch, available, expected):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(available)), raising=False)
    assert resolve_cpu_threads() == expected


def test_default_threads_fall_back_to_cpu_count(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert resolve_cpu_threads() == 6


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, "4"])
def test_invalid_cpu_threads_fail(threads):
    with pytest.raises(ValueError, match="cpu_threads"):
        training_environment(42, threads)


def test_explicit_threads_can_exceed_default_cap():
    assert resolve_cpu_threads(16) == 16
    environment = training_environment(42, 96)
    assert environment["OMP_NUM_THREADS"] == "96"
    assert environment["NUMEXPR_NUM_THREADS"] == environment["NUMEXPR_MAX_THREADS"] == "96"
