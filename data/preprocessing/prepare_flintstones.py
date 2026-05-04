"""FlintstonesSV preprocessing — wraps AR-LDM's flintstones_hdf5.py."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--arldm_repo", default="third_party/ARLDM")
    args = ap.parse_args()

    script = Path(args.arldm_repo) / "data_script" / "flintstones_hdf5.py"
    if not script.exists():
        raise FileNotFoundError(f"{script} not found. Clone AR-LDM first.")

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "python", str(script),
        "--data_dir", args.data_dir,
        "--save_path", args.save_path,
    ], check=True)


if __name__ == "__main__":
    main()
