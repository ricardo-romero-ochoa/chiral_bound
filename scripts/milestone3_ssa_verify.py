#!/usr/bin/env python3
"""Exact time-inhomogeneous SSA consistency check for Milestone 3."""
from pathlib import Path
import numpy as np
import pandas as pd

from pvchiral import FrankRDMEParams
from pvchiral.ssa import simulate_exact_ssa_one_cell
from pvchiral.pathinfo import (
    bernoulli_kl,
    symmetric_pved_activity_kl_ceiling,
    wrong_probability_lower_from_reverse_kl,
)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "milestone3_exact_ssa.csv"
RAW = ROOT / "data" / "milestone3_exact_ssa_realizations.npz"

p = FrankRDMEParams(omega=20.0, alpha=0.5, D=0.5)
g = 0.02
tau = 25.0
nx = 2
nreal = 200
seed = 180001

o = simulate_exact_ssa_one_cell(g, tau, params=p, nx=nx, nreal=nreal, seed=seed)
pg = float(np.mean(o["event_g"]))
p0 = float(np.mean(o["event_0"]))
Dh = float(np.mean(o["klint_g0"]))
D0 = float(np.mean(o["klint_0g"]))
Ah = float(np.mean(o["Achi_g"]))
A0 = float(np.mean(o["Achi_0"]))
Ch = symmetric_pved_activity_kl_ceiling(g, Ah, 2.0)
C0 = symmetric_pved_activity_kl_ceiling(g, A0, 2.0)

row = dict(
    g=g, tau_Q=tau, nx=nx, nreal=nreal, omega=p.omega, D=p.D, seed=seed,
    p_g=pg, p_0=p0, wrong_g=1 - pg,
    D_bern_empirical_g0=bernoulli_kl(pg, p0),
    D_bern_empirical_0g=bernoulli_kl(p0, pg),
    D_bern_Z2_0g=bernoulli_kl(0.5, pg),
    D_path_g0=Dh,
    D_path_g0_se=float(np.std(o["klint_g0"], ddof=1) / np.sqrt(nreal)),
    D_path_0g=D0,
    D_path_0g_se=float(np.std(o["klint_0g"], ddof=1) / np.sqrt(nreal)),
    realized_loglr_g0=float(np.mean(o["loglr_g0"])),
    realized_loglr_0g=float(np.mean(o["loglr_0g"])),
    Achi_g=Ah,
    Achi_g_se=float(np.std(o["Achi_g"], ddof=1) / np.sqrt(nreal)),
    Achi_0=A0,
    Achi_0_se=float(np.std(o["Achi_0"], ddof=1) / np.sqrt(nreal)),
    C_activity_g0=Ch, C_activity_0g=C0,
    wrong_lower_reverse_activity=wrong_probability_lower_from_reverse_kl(C0),
    ratio_Dber_empirical_Dpath_g0=(bernoulli_kl(pg, p0) / Dh if Dh > 0 else np.nan),
    ratio_Dber_Z2_Dpath_0g=(bernoulli_kl(0.5, pg) / D0 if D0 > 0 else np.nan),
    ratio_Dpath0g_C0=(D0 / C0 if C0 > 0 else np.nan),
    mean_events_g=float(np.mean(o["events_g"])), mean_events_0=float(np.mean(o["events_0"])),
)
pd.DataFrame([row]).to_csv(CSV, index=False)
np.savez_compressed(
    RAW,
    event_g=o["event_g"], event_0=o["event_0"],
    klint_g0=o["klint_g0"], klint_0g=o["klint_0g"],
    Achi_g=o["Achi_g"], Achi_0=o["Achi_0"],
    events_g=o["events_g"], events_0=o["events_0"],
    loglr_g0=o["loglr_g0"], loglr_0g=o["loglr_0g"],
)
print(pd.DataFrame([row]).T)
print(f"wrote {CSV}")
print(f"wrote {RAW}")
