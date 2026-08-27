#!/usr/bin/env python3
"""Regenerate the Milestone-3 production scan and archive realization-level data.

The production design is intentionally fixed to the deposited data:
- 1600 realizations for (tau_Q, g) = (100, 0.005)
- 600 realizations for each of the other six conditions
for 5200 realizations total.

Realization-level final events, path KL integrals, and chirality-weighted
activities are stored in ``data/milestone3_realizations.npz`` so that the
archived bootstrap confidence intervals
can be reproduced without rerunning the stochastic simulation.
"""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from pvchiral import FrankRDMEParams, simulate_postfreeze_branches
from pvchiral.pathinfo import (
    bernoulli_kl,
    bernoulli_probability_interval,
    wrong_probability_lower_from_reverse_kl,
    symmetric_pved_activity_kl_ceiling,
    symmetric_pved_weak_kl,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "milestone3_pathinfo.csv"
RAW = ROOT / "data" / "milestone3_realizations.npz"
DTC = ROOT / "data" / "milestone3_dt_convergence.csv"

PRODUCTION = [
    # case, tau_Q, g, nreal, seed
    (0, 100.0, 0.005, 1600, 160000),
    (1, 100.0, 0.010,  600, 160101),
    (2, 100.0, 0.015,  600, 160202),
    (3, 100.0, 0.020,  600, 160303),
    (4, 200.0, 0.005,  600, 160404),
    (5, 200.0, 0.010,  600, 160505),
    (6, 200.0, 0.015,  600, 160606),
]

DT_CONVERGENCE = [
    # case, tau_Q, g, nreal, dt, seed
    (0, 100.0, 0.015, 500, 0.020, 170000),
    (1, 100.0, 0.015, 500, 0.010, 170101),
    (2, 100.0, 0.015, 500, 0.005, 170202),
]


def _se(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def summarize_output(o: dict) -> dict:
    event_in = np.asarray(o["event_in"], dtype=float)
    event_on = np.asarray(o["event_on"], dtype=float)
    event_off = np.asarray(o["event_off"], dtype=float)
    kl_h0 = np.asarray(o["kl_h0"], dtype=float)
    kl_0h = np.asarray(o["kl_0h"], dtype=float)
    achi_h = np.asarray(o["b2_activity_h"], dtype=float)
    achi_0 = np.asarray(o["b2_activity_0"], dtype=float)

    n = len(event_on)
    pin = float(event_in.mean())
    pon = float(event_on.mean())
    p0 = float(event_off.mean())
    Dh = float(kl_h0.mean())
    Dr = float(kl_0h.mean())
    Ah = float(achi_h.mean())
    A0 = float(achi_0.mean())
    C_h = symmetric_pved_activity_kl_ceiling(o["g"], Ah, 2.0)
    C_0 = symmetric_pved_activity_kl_ceiling(o["g"], A0, 2.0)
    W_h = symmetric_pved_weak_kl(o["g"], Ah)
    W_0 = symmetric_pved_weak_kl(o["g"], A0)
    lo, hi = bernoulli_probability_interval(p0, Dh)
    qlower = wrong_probability_lower_from_reverse_kl(C_0)
    S = 0.5 * o["Rmax_block"] ** 2
    Demp_h0 = bernoulli_kl(pon, p0)
    Demp_0h = bernoulli_kl(p0, pon)
    Dz2_0h = bernoulli_kl(0.5, pon)

    return {
        "tau_Q": o["tau_Q"], "g": o["g"], "nx": o["nx"], "nreal": n,
        "dt": o["dt"], "block": o["block"], "a_hat": o["a_hat"],
        "t_hat": o["t_hat"], "Rmax_block": o["Rmax_block"], "S_gauss": S,
        "p_in": pin, "p_on": pon, "p_off": p0,
        "se_on": float(np.sqrt(max(pon * (1 - pon), 1e-15) / n)),
        "se_off": float(np.sqrt(max(p0 * (1 - p0), 1e-15) / n)),
        "wrong_on": 1 - pon,
        "D_bern_empirical_h0": Demp_h0,
        "D_bern_empirical_0h": Demp_0h,
        "D_bern_Z2_0h": Dz2_0h,
        "D_path_h0": Dh, "D_path_h0_se": _se(kl_h0),
        "D_path_0h": Dr, "D_path_0h_se": _se(kl_0h),
        "Achi_h": Ah, "Achi_h_se": _se(achi_h),
        "Achi_0": A0, "Achi_0_se": _se(achi_0),
        "C_activity_h0": C_h, "C_activity_0h": C_0,
        "Dweak_h0": W_h, "Dweak_0h": W_0,
        "event_upper_h0": hi, "event_lower_h0": lo,
        "wrong_lower_reverse_activity": qlower,
        "ratio_Dber_empirical_Dpath_h0": Demp_h0 / Dh if Dh > 0 else np.nan,
        "ratio_Dber_Z2_Dpath_0h": Dz2_0h / Dr if Dr > 0 else np.nan,
        "ratio_Dpath_S": Dh / S if S > 0 else np.nan,
        "ratio_Dpath_Dweak": Dh / W_h if W_h > 0 else np.nan,
        "ratio_Dpath_C": Dh / C_h if C_h > 0 else np.nan,
        "ratio_Dpath0h_C0": Dr / C_0 if C_0 > 0 else np.nan,
        "clips": int(o["clip_events_pre"] + o["clip_events_on"] + o["clip_events_off"]),
        "seed": o["seed"], "pre_g": o["pre_g"],
    }


def _run_production(args):
    case, tq, g, nreal, seed = args
    p = FrankRDMEParams(omega=100.0, alpha=0.5)
    nx = max(1, int(np.rint((p.D / (tq ** -0.5)) ** 0.5)))
    out = simulate_postfreeze_branches(
        g, tq, params=p, nx=nx, nreal=nreal, dt=0.01,
        a_i=-0.20, burn_time=10.0, seed=seed, pre_g=0.0,
    )
    return case, out


def _run_dt(args):
    case, tq, g, nreal, dt, seed = args
    p = FrankRDMEParams(omega=100.0, alpha=0.5)
    nx = max(1, int(np.rint((p.D / (tq ** -0.5)) ** 0.5)))
    out = simulate_postfreeze_branches(
        g, tq, params=p, nx=nx, nreal=nreal, dt=dt,
        a_i=-0.20, burn_time=10.0, seed=seed, pre_g=0.0,
    )
    return case, out


def _save_realizations(outputs):
    payload = {
        "tau_Q": np.array([PRODUCTION[i][1] for i, _ in outputs], dtype=float),
        "g": np.array([PRODUCTION[i][2] for i, _ in outputs], dtype=float),
        "nreal": np.array([PRODUCTION[i][3] for i, _ in outputs], dtype=int),
        "seed": np.array([PRODUCTION[i][4] for i, _ in outputs], dtype=int),
    }
    fields = [
        "event_in", "event_on", "event_off", "kl_h0", "kl_0h",
        "b2_activity_h", "b2_activity_0", "incoming_eta", "eta_on", "eta_off",
    ]
    for case, out in outputs:
        for field in fields:
            payload[f"case{case}_{field}"] = np.asarray(out[field])
    np.savez_compressed(RAW, **payload)


def main():
    with ProcessPoolExecutor(max_workers=7) as ex:
        outputs = list(ex.map(_run_production, PRODUCTION))
    outputs.sort(key=lambda x: x[0])
    rows = [summarize_output(out) for _, out in outputs]
    pd.DataFrame(rows).to_csv(OUT, index=False)
    _save_realizations(outputs)

    for r in rows:
        print(
            f"tau={r['tau_Q']:g} g={r['g']:.3f} n={r['nreal']} "
            f"p_on={r['p_on']:.4f} p0={r['p_off']:.4f} "
            f"D0h={r['D_path_0h']:.4f} qmin={r['wrong_lower_reverse_activity']:.4f}"
        )

    with ProcessPoolExecutor(max_workers=3) as ex:
        dt_outputs = list(ex.map(_run_dt, DT_CONVERGENCE))
    dt_outputs.sort(key=lambda x: x[0])
    drows = [summarize_output(out) for _, out in dt_outputs]
    pd.DataFrame(drows).to_csv(DTC, index=False)

    print(f"wrote {OUT}")
    print(f"wrote {RAW}")
    print(f"wrote {DTC}")


if __name__ == "__main__":
    main()
