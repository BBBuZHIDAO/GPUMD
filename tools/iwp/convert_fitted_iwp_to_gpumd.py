#!/usr/bin/env python3
"""Convert the four fitted Mo-Mo/Mo-S/Mo-Se/S-Se IWP rows to GPUMD format."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


PAIR_MATRIX = (
    0, 1, 2,  # Mo-Mo, Mo-S, Mo-Se
    1, 3, 3,  # S-Mo, S-S fallback, S-Se
    2, 3, 3,  # Se-Mo, Se-S, Se-Se fallback
)


def read_parameters(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(token) for token in line.split()]
        if len(values) != 9:
            raise ValueError(f"{path}:{line_number}: expected 9 columns, got {len(values)}")
        rows.append(values)
    if len(rows) != 4:
        raise ValueError(f"{path}: expected exactly 4 numeric rows, got {len(rows)}")
    return rows


def format_row(values: Iterable[float]) -> str:
    return " ".join(f"{value:.10f}" for value in values)


def write_gpumd(
    output: Path,
    fitted: list[list[float]],
    wire_group_method: int,
    rcut_normal: float,
    rcut_global: float,
) -> None:
    # Keep the numerical potential file comment-free: the current GPUMD
    # constructor reads it with fscanf and therefore does not skip # lines.
    lines = [
        "nep_iwp 3 Mo S Se",
        str(wire_group_method),
    ]
    for fitted_index in PAIR_MATRIX:
        lines.append(format_row([*fitted[fitted_index], 1.0, rcut_normal, rcut_global]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="4x9 fitted_IWP_parameters.dat")
    parser.add_argument("output", type=Path, help="output GPUMD IWP potential file")
    parser.add_argument("--wire-group-method", type=int, default=0)
    parser.add_argument("--rcut-normal", type=float, default=6.0,
                        help="same-wire unlike-atom search cutoff for the nearest four normal neighbors")
    parser.add_argument("--rcut-global", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wire_group_method < 0:
        raise ValueError("wire group method must be non-negative")
    if args.rcut_normal <= 0.0 or args.rcut_global <= 0.0:
        raise ValueError("cutoffs must be positive")
    write_gpumd(
        args.output,
        read_parameters(args.input),
        args.wire_group_method,
        args.rcut_normal,
        args.rcut_global,
    )
    print(f"Wrote GPUMD IWP potential: {args.output}")


if __name__ == "__main__":
    main()
