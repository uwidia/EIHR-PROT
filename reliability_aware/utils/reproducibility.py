"""Opt-in reproducibility controls for the confidence-gate ablation suite."""
from __future__ import annotations

import os
import random
import sys

from reliability_aware.utils.initialization import resolve_initialization_seed


def validate_training_seed(seed):
    if seed is None or seed == "random":
        raise ValueError("training_seed must be a resolved integer shared across variants")
    return resolve_initialization_seed(seed)


def resolve_cpu_threads(value=None):
    """Use a modest parallel default; freeze the resolved count in suite configs."""
    if value is None:
        try:
            available = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available = os.cpu_count() or 1
        return max(1, min(8, available))
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("cpu_threads must be a positive integer")
    return value


def training_environment(seed, cpu_threads=None):
    seed = validate_training_seed(seed)
    threads = str(resolve_cpu_threads(cpu_threads))
    return {
        "PYTHONHASHSEED": str(seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NVIDIA_TF32_OVERRIDE": "0",
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "NUMEXPR_MAX_THREADS": threads,
    }


def prepare_training_process(seed, cpu_threads=None):
    """Call before importing torch/numpy. Restart to apply startup-only settings.

    This also covers users running an individual generated training command,
    without requiring shell-specific environment-variable syntax.
    """
    required = training_environment(seed, cpu_threads)
    if any(os.environ.get(key) != value for key, value in required.items()):
        environment = dict(os.environ, **required)
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def seed_worker(worker_id):
    """Use each loader's generator-derived worker seed for all three RNGs."""
    import numpy as np
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_training_rng(seed, train_loader=None, val_loader=None):
    import numpy as np
    import torch

    seed = validate_training_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for offset, loader in enumerate((train_loader, val_loader)):
        if loader is None:
            continue
        sampler = getattr(loader, "batch_sampler", None)
        if hasattr(sampler, "seed"):
            sampler.seed = seed
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)
        # Separate streams keep iterator/worker creation out of the dropout RNG.
        loader.generator = torch.Generator().manual_seed((seed + offset) % 2**32)
        loader.worker_init_fn = seed_worker
        if getattr(loader, "persistent_workers", False):
            raise ValueError("Reproducible runs require persistent_workers=False so worker seeds reset")


def configure_training(seed, train_loader=None, val_loader=None, *, cpu_threads=None):
    """Set deterministic kernels and reset RNGs; return checkpoint provenance."""
    import torch

    seed = validate_training_seed(seed)
    required = training_environment(seed, cpu_threads)
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise RuntimeError("Start seeded training through run_model_training.py to apply the reproducible process environment")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cpu_threads = resolve_cpu_threads(cpu_threads)
    torch.set_num_threads(cpu_threads)
    if torch.get_num_interop_threads() != cpu_threads:
        torch.set_num_interop_threads(cpu_threads)
    seed_training_rng(seed, train_loader, val_loader)
    return {
        "protocol": "confidence_gate_reproducibility_v2",
        "training_seed": seed,
        "environment": required,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "tf32": False,
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "loader_seeds": {"train": seed, "validation": (seed + 1) % 2**32},
        "torch_version": str(torch.__version__),
        "numpy_version": __import__("numpy").__version__,
        "python_version": sys.version,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
