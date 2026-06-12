import argparse
import sys


def _check_torch_import():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime helper
        msg = (
            "Failed to import PyTorch: {}\n".format(exc)
            + "Common causes on Windows: missing Microsoft Visual C++ Redistributable,\n"
            + "or a CUDA-enabled wheel installed on a machine without matching drivers.\n\n"
            + "Quick fixes:\n"
            + "  - Install Microsoft Visual C++ Redistributable (2015-2022) x64: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist\n"
            + "  - Recreate the CPU environment with `uv sync --extra cpu`.\n"
            + "  - Recreate the CUDA environment with `uv sync --extra cu128`.\n\n"
            + "Verify with the matching `uv run --extra cpu ...` or "
            + "`uv run --extra cu128 ...` command from the README.\n"
        )
        print(msg, file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESM embeddings into split-specific shard files."
    )
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--fasta_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--valid_hashes", type=str, default="hashlist.txt")
    parser.add_argument(
        "--manifest_filename",
        type=str,
        default=None,
        help="Manifest filename without .csv. Defaults to <split>_manifest.",
    )
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--toks_per_batch", type=int, default=4096)
    parser.add_argument("--truncation_seq_length", type=int, default=1022)
    parser.add_argument("--repr_layer", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    _check_torch_import()
    manifest_filename = args.manifest_filename or f"{args.split}_manifest"

    # Import the embedding extractor after the torch pre-check so we can
    # present a friendly error message if torch fails to import.
    import reliability_aware.utils.extract_esm_embeddings as extract_embeddings

    extract_embeddings.extract_fasta_embeddings(
        fasta_path=args.fasta_file,
        output_dir=args.outdir,
        valid_hashes_path=args.valid_hashes,
        manifest_filename=manifest_filename,
        model_name=args.model,
        toks_per_batch=args.toks_per_batch,
        truncation_seq_length=args.truncation_seq_length,
        repr_layer=args.repr_layer,
        shard_size=args.shard_size,
        use_fp16=args.use_fp16,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
    )


if __name__ == "__main__":
    main()
