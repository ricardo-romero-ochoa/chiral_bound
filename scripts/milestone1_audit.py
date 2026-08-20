#!/usr/bin/env python3
"""Numerical audit for the Milestone-1 paired-response theorem."""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np

from pvchiral import (
    Lambda,
    Theta_k,
    bound_T2,
    pved_activity_required_for_global_confidence,
    pved_response_activity,
)
from pvchiral.kz import R_for_global_confidence
from pvchiral.network import frank_chemostat_a, frank_network
from pvchiral.reduction import eps_eff

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "milestone1_frank_grid.csv"


def frank_grid():
    g = 1e-6
    rows = []
    for alpha, k3p, s_target, qfrac in itertools.product(
        (0.0, 0.5, 1.0), (1.5, 2.0, 3.0), (0.02, 0.04, 0.08), (0.1, 0.4, 0.8)
    ):
        Q = qfrac * ((1.0 + k3p) * s_target / 2.0) * s_target / 2.0
        a_chemo = frank_chemostat_a(s_target, Q, k3p)
        if a_chemo <= 0:
            continue
        net = frank_network(a_chemo, Q, k3p, alpha, g=g)
        B = np.array([r.B for r in net.rxns])
        al = np.array([r.alpha for r in net.rxns])
        Jp = np.array([r.Jp for r in net.rxns])
        Jm = np.array([r.Jm for r in net.rxns])
        Lresp = pved_response_activity(B, al, Jp, Jm, net.s)
        new_bound = g * np.sqrt(net.s * Theta_k(net) * Lresp)
        old_bound = bound_T2(net, g=g)
        eps = abs(eps_eff(net, g=g))
        rows.append(
            dict(
                alpha=alpha,
                k3p=k3p,
                s_target=s_target,
                qfrac=qfrac,
                epsilon=eps,
                old_bound=old_bound,
                paired_bound=new_bound,
                old_saturation=eps / old_bound,
                paired_saturation=eps / new_bound,
                Lambda=Lambda(net),
                Lambda_resp=Lresp,
                Lambda_resp_over_Lambda=Lresp / Lambda(net),
            )
        )
    return rows


def prebiotic_requirements():
    YEAR = 3.156e7
    g = 5e-17
    b = 1e-4
    rho = 6.022e23
    D = 1e-9
    krac = np.log(2.0) / (1e5 * YEAR)
    cases = {
        "racemization": 2 * krac,
        "annual": np.sqrt(2 * b / YEAR),
        "diurnal": np.sqrt(2 * b / 1e5),
    }
    out = []
    for name, ahat in cases.items():
        xi = np.sqrt(D / ahat)
        Nxi = rho * xi**3
        for M in (1, 100_000, 1_000_000_000):
            Rreq = R_for_global_confidence(M, 0.95)
            Lreq = pved_activity_required_for_global_confidence(
                g, ahat, Nxi, M, 0.95
            )
            out.append((name, M, ahat, xi, Nxi, Rreq, Lreq))
    return out


def main():
    rows = frank_grid()
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Frank grid: {len(rows)} operating points")
    print(
        "old saturation min/max:",
        min(r["old_saturation"] for r in rows),
        max(r["old_saturation"] for r in rows),
    )
    print(
        "paired saturation min/max:",
        min(r["paired_saturation"] for r in rows),
        max(r["paired_saturation"] for r in rows),
    )
    for alpha in (0.0, 0.5, 1.0):
        subset = [r for r in rows if r["alpha"] == alpha]
        print(
            f"alpha={alpha:g}: paired saturation",
            min(r["paired_saturation"] for r in subset),
            max(r["paired_saturation"] for r in subset),
            "Lambda_resp/Lambda",
            min(r["Lambda_resp_over_Lambda"] for r in subset),
            max(r["Lambda_resp_over_Lambda"] for r in subset),
        )

    print("\nNecessary convex-PVED Lambda for 95% all-favored confidence")
    print("case M R_required Lambda_required[s^-1]")
    for name, M, ahat, xi, Nxi, Rreq, Lreq in prebiotic_requirements():
        print(f"{name:12s} {M:10d} {Rreq:8.4f} {Lreq:.6e}")


if __name__ == "__main__":
    main()
