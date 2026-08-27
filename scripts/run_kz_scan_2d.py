#!/usr/bin/env python3
"""Generate or resume a compact two-dimensional consistency scan."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
from pvchiral.field2d import quench_observables_2d

THETA = 1e-6
TAUQS = [3e3, 6e3, 1.2e4]
MULTS = [0.25, 0.50, 1.0, 2.0, 4.0]
COEFF = 0.10
OUT = Path(__file__).resolve().parents[1] / "data" / "kz_scan_2d.csv"
FIELDS = ["dimension", "tau_Q", "eps", "f_L", "f_L_sem", "wall_density",
          "wall_density_sem", "nreal", "nx", "seed"]


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
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, choices=TAUQS)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    rows = [] if args.reset else read_rows()
    done = {(float(r["tau_Q"]), float(r["eps"])) for r in rows}
    selected = TAUQS if args.tau is None else [args.tau]
    for i, tq in enumerate(TAUQS):
        if tq not in selected:
            continue
        eps0 = COEFF * np.sqrt(THETA) * tq**-0.5
        for j, mult in enumerate(MULTS):
            eps = eps0*mult
            key = (float(tq), float(eps))
            if key in done:
                continue
            obs = quench_observables_2d(eps, tq, THETA, nx=32, nreal=8,
                                        a_i=-0.20, a_meas=0.03,
                                        seed=22100+100*i+j)
            row = {"dimension":2, "tau_Q":tq, "eps":eps,
                   "f_L":obs["f_L"], "f_L_sem":obs["f_L_sem"],
                   "wall_density":obs["wall_density"],
                   "wall_density_sem":obs["wall_density_sem"],
                   "nreal":obs["nreal"], "nx":obs["nx"],
                   "seed":22100+100*i+j}
            rows.append(row); done.add(key); write_rows(rows)
            print(f"tau_Q={tq:8.0f} eps={eps:10.3e} "
                  f"f_L={obs['f_L']:.4f}+/-{obs['f_L_sem']:.4f}", flush=True)

if __name__ == "__main__":
    main()
