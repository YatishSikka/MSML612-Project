"""AR-LDM baseline wrapper.

AR-LDM uses its own training/inference code. The cleanest approach is:
  1. Clone github.com/xichenpan/ARLDM into `third_party/ARLDM`
  2. Run its inference script directly for baseline numbers
  3. Point evaluation at its output dir

This file provides a subprocess-based wrapper so our evaluation harness can
trigger AR-LDM generation uniformly with our own pipeline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List


def run_arldm_inference(arldm_repo: str, config: str, ckpt: str,
                        output_dir: str) -> List[Path]:
    cmd = [
        "python", f"{arldm_repo}/main.py",
        "--config", config,
        "--mode", "sample",
        "--ckpt", ckpt,
        "--output_dir", output_dir,
    ]
    subprocess.run(cmd, check=True)
    return sorted(Path(output_dir).glob("*.png"))
