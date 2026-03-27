"""Collect calibration data from VizDoom for full-integer INT8 quantization.

Runs in the training venv (which has vizdoom). Outputs a .npz file that
can be consumed by the export pipeline in the export venv.

Usage:
    uv run python -m training.calibrate
    uv run python -m training.calibrate --cfg training/scenarios/basic.cfg -n 500
"""

import argparse
from pathlib import Path

from training.export import (
    collect_representative_dataset,
    save_calibration_data,
)


def main():
    parser = argparse.ArgumentParser(
        description="Collect calibration data for INT8 quantization",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="VizDoom config file",
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=200,
        help="Number of calibration frames to collect",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="models/calibration_data.npz",
        help="Output .npz file path",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    samples = collect_representative_dataset(
        cfg_path=args.cfg,
        n_samples=args.samples,
    )
    save_calibration_data(samples, out)


if __name__ == "__main__":
    main()
