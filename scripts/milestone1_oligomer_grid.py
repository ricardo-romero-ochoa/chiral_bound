#!/usr/bin/env python3
"""Second-topology audit of the paired-response theorem.

The initial saturation audit used the Frank network alone,
which cannot distinguish a property of the pairing step from a property of that
network.  This script repeats the audit on a structurally different topology --
a monomer/dimer network with dimerisation (B_j = 0) and dimer epimerisation
(|B_j| = 2) channels absent from the Frank model -- over an independent grid of
driven operating points.
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np

from pvchiral import Lambda, Theta_k, bound_T2, pved_response_activity
from pvchiral.network import oligomer_network
from pvchiral.reduction import eps_eff

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "milestone1_oligomer_grid.csv"
G = 1e-6


def oligomer_grid():
    rows = []
    for alpha, A, kw, W in itertools.product(
        (0.0, 0.5, 1.0), (0.02, 0.05, 0.10), (0.1, 0.4, 1.0), (1e-5, 1e-4, 1e-3)
    ):
        net = oligomer_network(alpha=alpha, g=G, A=A, W=W, kw=kw)
        if net.s < 1e-8:
            continue
        B = np.array([r.B for r in net.rxns])
        al = np.array([r.alpha for r in net.rxns])
        Jp = np.array([r.Jp for r in net.rxns])
        Jm = np.array([r.Jm for r in net.rxns])
        Lresp = pved_response_activity(B, al, Jp, Jm, net.s)
        paired = G * np.sqrt(net.s * Theta_k(net) * Lresp)
        old = bound_T2(net, g=G)
        eps = abs(eps_eff(net, g=G))
        if paired <= 0 or old <= 0:
            continue
        rows.append(dict(alpha=alpha, A_chemostat=A, kw=kw, W_chemostat=W,
                         epsilon=eps, old_bound=old, paired_bound=paired,
                         old_saturation=eps / old, paired_saturation=eps / paired,
                         Lambda=Lambda(net), Lambda_resp=Lresp,
                         Lambda_resp_over_Lambda=Lresp / Lambda(net)))
    return rows


def main():
    rows = oligomer_grid()
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ps = np.array([r["paired_saturation"] for r in rows])
    os_ = np.array([r["old_saturation"] for r in rows])
    print(f"{len(rows)} operating points")
    print(f"  paired saturation : {ps.min():.4f} - {ps.max():.4f}")
    print(f"  old envelope      : {os_.min():.4f} - {os_.max():.4f}")
    assert np.all(ps <= 1.0 + 1e-9), "paired bound violated"
    assert np.all(os_ <= 1.0 + 1e-9), "old bound violated"
    print("  no violations")


if __name__ == "__main__":
    main()
