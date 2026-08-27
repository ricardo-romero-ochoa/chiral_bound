#!/usr/bin/env python3
"""Bootstrap uncertainty for the Milestone-3 finite-time production data.

Resamples realization indices with replacement within each condition. The same
index is applied to the biased/zero-field branches and to the path/activity
observables so the common incoming-state pairing is preserved. Percentile 95%
intervals are reported from 20,000 resamples with a fixed RNG seed.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from pvchiral.pathinfo import bernoulli_kl, symmetric_pved_activity_kl_ceiling, wrong_probability_lower_from_reverse_kl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "milestone3_realizations.npz"
RAW_SSA = ROOT / "data" / "milestone3_exact_ssa_realizations.npz"
OUT = ROOT / "data" / "milestone3_bootstrap.csv"
OUT_SSA = ROOT / "data" / "milestone3_exact_ssa_bootstrap.csv"
NBOOT = 20_000
SEED = 171717


def _z2_bernoulli_vec(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return 0.5 * np.log(0.5 / p) + 0.5 * np.log(0.5 / (1 - p))


def _empirical_bernoulli_vec(p0, p1):
    p0 = np.clip(np.asarray(p0, dtype=float), 1e-15, 1 - 1e-15)
    p1 = np.clip(np.asarray(p1, dtype=float), 1e-15, 1 - 1e-15)
    return p0 * np.log(p0 / p1) + (1 - p0) * np.log((1 - p0) / (1 - p1))


def _qmin_vec(C):
    C = np.asarray(C, dtype=float)
    return 0.5 * (1 - np.sqrt(np.maximum(0.0, 1 - np.exp(-2 * C))))


def _interval(x):
    lo, hi = np.percentile(np.asarray(x, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_condition(rng, event_on, event_off, kl_0h, achi_0, g):
    n = len(event_on)
    out = {k: np.empty(NBOOT, dtype=float) for k in [
        "p_on", "q", "D_bern_empirical_0h", "D_bern_Z2_0h",
        "D_path_0h", "C_activity_0h", "qmin", "ratio_Z2_Dpath", "ratio_Dpath_C",
    ]}
    pos = 0
    chunk = 500
    while pos < NBOOT:
        m = min(chunk, NBOOT - pos)
        idx = rng.integers(0, n, size=(m, n))
        pon = event_on[idx].mean(axis=1)
        poff = event_off[idx].mean(axis=1)
        dpath = kl_0h[idx].mean(axis=1)
        a0 = achi_0[idx].mean(axis=1)
        dz2 = _z2_bernoulli_vec(pon)
        demp = _empirical_bernoulli_vec(poff, pon)
        C0 = (g * g / 8.0) * np.exp(abs(g)) * a0  # Bmax=2 for the Frank network
        qmin = _qmin_vec(C0)
        sl = slice(pos, pos + m)
        out["p_on"][sl] = pon
        out["q"][sl] = 1 - pon
        out["D_bern_empirical_0h"][sl] = demp
        out["D_bern_Z2_0h"][sl] = dz2
        out["D_path_0h"][sl] = dpath
        out["C_activity_0h"][sl] = C0
        out["qmin"][sl] = qmin
        out["ratio_Z2_Dpath"][sl] = dz2 / dpath
        out["ratio_Dpath_C"][sl] = dpath / C0
        pos += m
    return out


def main():
    rng = np.random.default_rng(SEED)
    raw = np.load(RAW)
    path = pd.read_csv(ROOT / "data" / "milestone3_pathinfo.csv")
    rows = []
    for i, rec in path.iterrows():
        event_on = raw[f"case{i}_event_on"].astype(float)
        event_off = raw[f"case{i}_event_off"].astype(float)
        kl_0h = raw[f"case{i}_kl_0h"].astype(float)
        achi_0 = raw[f"case{i}_b2_activity_0"].astype(float)
        b = _bootstrap_condition(rng, event_on, event_off, kl_0h, achi_0, float(rec.g))
        row = {"tau_Q": float(rec.tau_Q), "g": float(rec.g), "nreal": int(rec.nreal), "nboot": NBOOT, "bootstrap_seed": SEED}
        for key, arr in b.items():
            lo, hi = _interval(arr)
            row[f"{key}_lo"] = lo
            row[f"{key}_hi"] = hi
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT, index=False)

    sraw = np.load(RAW_SSA)
    srec = pd.read_csv(ROOT / "data" / "milestone3_exact_ssa.csv").iloc[0]
    b = _bootstrap_condition(
        rng,
        sraw["event_g"].astype(float), sraw["event_0"].astype(float),
        sraw["klint_0g"].astype(float), sraw["Achi_0"].astype(float), float(srec.g),
    )
    srow = {"g": float(srec.g), "tau_Q": float(srec.tau_Q), "nreal": int(srec.nreal), "nboot": NBOOT, "bootstrap_seed": SEED}
    # Rename main-scan notation to the SSA notation only in the output file.
    for key, arr in b.items():
        lo, hi = _interval(arr)
        srow[f"{key}_lo"] = lo
        srow[f"{key}_hi"] = hi
    pd.DataFrame([srow]).to_csv(OUT_SSA, index=False)
    print(f"wrote {OUT}")
    print(f"wrote {OUT_SSA}")


if __name__ == "__main__":
    main()
