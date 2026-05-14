import argparse
import utils.extract_esm_embeddings as extract_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--valid_hashes", type=str, required=True)
    parser.add_argument("--manifest_filename", type=str, default="manifest")
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--toks_per_batch", type=int, default=4096)
    parser.add_argument("--truncation_seq_length", type=int, default=1022)
    parser.add_argument("--repr_layer", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    extract_embeddings.extract_fasta_embeddings(
        fasta_path=args.fasta_file,
        output_dir=args.outdir,
        valid_hashes_path=args.valid_hashes,
        manifest_filename=args.manifest_filename,
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
