"""FID between two image directories using pytorch-fid."""

from __future__ import annotations

import argparse


def compute_fid(real_dir: str, gen_dir: str, device: str = "cuda",
                batch_size: int = 50, dims: int = 2048) -> float:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    return calculate_fid_given_paths(
        [real_dir, gen_dir], batch_size=batch_size, device=device, dims=dims,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--batch_size", type=int, default=50)
    args = ap.parse_args()
    print(f"FID: {compute_fid(args.real, args.gen, batch_size=args.batch_size):.4f}")
