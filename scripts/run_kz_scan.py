#!/usr/bin/env python3
"""Generate or resume the one-dimensional KZ scan with realization-level errors."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pvchiral.field1d import (
    bootstrap_exponent_from_scan,
    exponent_from_scan,
    quench_observables,
    xi_hat_over_dx,
)

THETA = 1e-6
TAUQS = [3e3, 6e3, 1.2e4, 2.4e4, 4.8e4]
MULTS = [0.35, 0.50, 0.70, 1.0, 1.4, 2.0]
OUT = Path(__file__).resolve().parents[1] / "data" / "kz_scan_1d.csv"
FIELDS = [
    "dimension", "tau_Q", "eps", "f_L", "f_L_sem", "wall_density",
    "wall_density_sem", "nreal", "nx", "xi_over_dx", "seed",
]


def read_rows():
    if not OUT.exists():
        return []
    with OUT.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(rows):
    rows = sorted(rows, key=lambda r: (float(r["tau_Q"]), float(r["eps"])))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def generate(selected_tau=None, reset=False):
    rows = [] if reset else read_rows()
    done = {(float(r["tau_Q"]), float(r["eps"])) for r in rows}
    tau_values = TAUQS if selected_tau is None else [selected_tau]
    for i, tq in enumerate(TAUQS):
        if tq not in tau_values:
            continue
        eps0 = 0.5 * np.sqrt(THETA) * tq ** (-3.0 / 8.0)
        for j, mult in enumerate(MULTS):
            eps = eps0 * mult
            key = (float(tq), float(eps))
            if key in done:
                continue
            obs = quench_observables(
                eps,
                tq,
                THETA,
                nx=128,
                nreal=24,
                a_i=-0.20,
                a_meas=0.03,
                seed=11235 + 100 * i + j,
            )
            row = {
                "dimension": 1,
                "tau_Q": tq,
                "eps": eps,
                "f_L": obs["f_L"],
                "f_L_sem": obs["f_L_sem"],
                "wall_density": obs["wall_density"],
                "wall_density_sem": obs["wall_density_sem"],
                "nreal": obs["nreal"],
                "nx": obs["nx"],
                "xi_over_dx": xi_hat_over_dx(tq),
                "seed": 11235 + 100 * i + j,
            }
            rows.append(row)
            done.add(key)
            write_rows(rows)
            print(
                f"tau_Q={tq:8.0f} eps={eps:10.3e} "
                f"f_L={obs['f_L']:.4f}+/-{obs['f_L_sem']:.4f}",
                flush=True,
            )
    return rows


def summarize(rows):
    if len(rows) < 18:
        return
    fit = exponent_from_scan(
        [float(r["tau_Q"]) for r in rows],
        [float(r["eps"]) for r in rows],
        [float(r["f_L"]) for r in rows],
    )
    boot = bootstrap_exponent_from_scan(
        [float(r["tau_Q"]) for r in rows],
        [float(r["eps"]) for r in rows],
        [float(r["f_L"]) for r in rows],
        [float(r["f_L_sem"]) for r in rows],
        n_boot=1000,
    )
    print(
        f"slope={fit['slope']:+.4f}, OLS 95% CI={fit['slope_ci95']}, "
        f"bootstrap 95% CI={boot['slope_ci95']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, choices=TAUQS)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    rows = generate(args.tau, args.reset)
    summarize(rows)


if __name__ == "__main__":
    main()
