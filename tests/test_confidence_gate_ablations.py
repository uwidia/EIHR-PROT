import json
from pathlib import Path

import pytest
import torch
import yaml

from models.sequence_homology_ablation import (
    CONFIDENCE_GATE_FEATURES,
    build_sequence_homology_confidence_gate_model as build_model,
)
from reliability_aware.utils.initialization import resolve_initialization_seed
from reliability_aware.utils.reproducibility import resolve_cpu_threads
from reliability_aware.utils.inference_runtime import load_model_from_checkpoint
from scripts.run_confidence_gate_ablations import prepare_suite, VARIANTS


HPARAMS = dict(learning_rate=0.001, attn_hidden_dim=4, head_hidden_dim=4,
               gate_hidden_dim=4, attn_dropout=0, head_dropout=0, gate_dropout=0,
               pos_weight_cap=5, lambda_hier=0)


def build(omitted=None, seed=42):
    hp = dict(HPARAMS, omitted_gate_feature=omitted, initialization_seed=seed)
    model, optimizer = build_model(hp, ["GO:1", "GO:2"], "cpu")
    return model, optimizer, hp


def inputs():
    return dict(padded=torch.randn(2, 3, 1280), mask=torch.ones(2, 3, dtype=torch.bool),
                homology_scores=torch.tensor([[0.2, 0.9], [0., 0.]]),
                gate_features=torch.tensor([[75., .8, 1.6, 1.], [0., 0., 0., 0.]]))


@pytest.mark.parametrize("omitted", [None, *CONFIDENCE_GATE_FEATURES])
def test_selection_precedes_layernorm_and_omitted_signal_has_no_effect(omitted):
    model, optimizer, _ = build(omitted)
    batch = inputs()
    original = {k: v.clone() for k, v in batch.items()}
    width = 4 if omitted is None else 3
    assert model.gate[0].normalized_shape == (width,)
    assert model.gate[1].in_features == width
    seen = []
    hook = model.gate[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    # Neutral initialization alone would hide broken feature selection.
    with torch.no_grad():
        model.gate[-2].weight.copy_(torch.tensor([[1., -1., .5, .3], [-.2, .5, 1., -.8]]))
    model.eval()
    out = model(**batch)
    indices = [i for i, name in enumerate(CONFIDENCE_GATE_FEATURES) if name != omitted]
    assert torch.equal(seen[0], batch["gate_features"][:, indices])
    assert torch.equal(out["homology_scores"], original["homology_scores"])
    if omitted is not None:
        changed = dict(batch, gate_features=batch["gate_features"].clone())
        changed["gate_features"][:, CONFIDENCE_GATE_FEATURES.index(omitted)] = float("nan")
        other = model(**changed)
        assert torch.equal(out["gate_weights"], other["gate_weights"])
        assert torch.equal(out["probs"], other["probs"])
    assert all(torch.equal(batch[k], original[k]) for k in batch)
    loss = torch.nn.functional.binary_cross_entropy(out["probs"], torch.tensor([[1., 0.], [0., 1.]]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert torch.isfinite(model(**batch)["probs"]).all()
    hook.remove()


def test_fixed_initialization_repeatability_shared_branches_and_rng_isolation():
    before = torch.random.get_rng_state().clone()
    full, _, _ = build()
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(37)
    repeated, _, _ = build()
    assert all(torch.equal(p, repeated.state_dict()[k]) for k, p in full.state_dict().items())
    for omitted in CONFIDENCE_GATE_FEATURES:
        ablated, _, _ = build(omitted)
        for name, value in full.state_dict().items():
            if name.startswith(("seq_branch.", "head.")):
                assert torch.equal(value, ablated.state_dict()[name])
        assert torch.equal(ablated.gate[-2].weight, torch.zeros_like(ablated.gate[-2].weight))
    different, _, _ = build(seed=43)
    assert not torch.equal(full.head.net[2].weight, different.head.net[2].weight)


def test_random_seed_is_recorded_and_replays():
    model, _, hp = build(seed="random")
    assert isinstance(hp["initialization_seed"], int)
    replay, _, _ = build(seed=hp["initialization_seed"])
    assert all(torch.equal(p, replay.state_dict()[k]) for k, p in model.state_dict().items())


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "42", "typo"])
def test_invalid_seeds_fail(seed):
    with pytest.raises(ValueError, match="initialization_seed"):
        resolve_initialization_seed(seed)


def test_invalid_feature_fails():
    with pytest.raises(ValueError, match="Unknown omitted"):
        build("typo")


@pytest.mark.parametrize("omitted", [None, *CONFIDENCE_GATE_FEATURES])
def test_inference_checkpoint_roundtrip(tmp_path, omitted):
    model, _, hp = build(omitted)
    model.eval()
    # Also cover legacy checkpoints, which have neither new setting.
    if omitted is None:
        hp.pop("initialization_seed")
        hp.pop("omitted_gate_feature")
    path = tmp_path / "best_model.pt"
    torch.save(dict(hparams=hp, model_state_dict=model.state_dict()), path)
    restored, _ = load_model_from_checkpoint(model_builder=build_model, checkpoint_path=path,
                                              go_terms=["GO:1", "GO:2"], device=torch.device("cpu"))
    restored.eval()
    batch = inputs()
    assert torch.equal(model(**batch)["probs"], restored(**batch)["probs"])


def test_suite_freezes_selected_parameters_and_preserves_source(tmp_path):
    source = Path("configs/sequence_homology_confidence_gate.yaml")
    original = source.read_bytes()
    selected = yaml.safe_load(original)
    out = tmp_path / "experiment"
    manifest = prepare_suite(source, out, 42)
    assert source.read_bytes() == original
    assert len(manifest["jobs"]) == 15
    assert json.loads((out / "suite.json").read_text()) == manifest
    for variant, omitted in VARIANTS.items():
        config = yaml.safe_load((out / "configs" / f"{variant}.yaml").read_text())
        assert config["use_search_results"] is False
        assert Path(config["base_dir_final"]).is_relative_to(out)
        for key in ("final_epochs", "patience", "batch_size"):
            assert config[key] == selected[key]
        for aspect, hp in config["promising_hparams"].items():
            assert hp.pop("initialization_seed") == 42
            assert hp.pop("training_seed") == 42
            assert hp.pop("reproducible_training") is True
            assert hp.pop("cpu_threads") == manifest["cpu_threads"] == resolve_cpu_threads()
            assert hp.pop("omitted_gate_feature") == omitted
            assert hp == selected["promising_hparams"][aspect]
    with pytest.raises(FileExistsError):
        prepare_suite(source, out, 42)


def test_random_suite_draws_one_shared_seed(tmp_path):
    manifest = prepare_suite("configs/sequence_homology_confidence_gate.yaml", tmp_path / "suite", "random", ["BP"])
    assert len(manifest["jobs"]) == 5
    assert {j["hparams"]["initialization_seed"] for j in manifest["jobs"]} == {manifest["initialization_seed"]}

    assert {j["hparams"]["training_seed"] for j in manifest["jobs"]} == {manifest["initialization_seed"]}
    assert manifest["environment"]["PYTHONHASHSEED"] == str(manifest["training_seed"])


def test_cli_freezes_explicit_cpu_threads_for_all_variants(tmp_path, monkeypatch):
    from scripts.run_confidence_gate_ablations import main
    out = tmp_path / "threaded_suite"
    monkeypatch.setattr("sys.argv", ["run_confidence_gate_ablations.py", "--seed", "42",
                                    "--output-dir", str(out), "--cpu-threads", "4"])
    main()
    manifest = json.loads((out / "suite.json").read_text())
    assert manifest["cpu_threads"] == 4
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert manifest["environment"][key] == "4"
    for path in (out / "configs").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        assert {hp["cpu_threads"] for hp in config["promising_hparams"].values()} == {4}
    assert {job["hparams"]["cpu_threads"] for job in manifest["jobs"]} == {4}


def test_invalid_thread_count_creates_no_suite(tmp_path):
    out = tmp_path / "invalid_suite"
    with pytest.raises(ValueError, match="cpu_threads"):
        prepare_suite("configs/sequence_homology_confidence_gate.yaml", out, 42, cpu_threads=0)
    assert not out.exists()
