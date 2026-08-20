#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "milestone3_pathinfo.csv"
BOOT = ROOT / "data" / "milestone3_bootstrap.csv"
DTC = ROOT / "data" / "milestone3_dt_convergence.csv"


def main():
    d = pd.read_csv(DATA)
    assert np.all(d.D_bern_empirical_h0 <= d.D_path_h0 + 1e-12)
    assert np.all(d.D_bern_empirical_0h <= d.D_path_0h + 1e-12)
    assert np.all(d.D_bern_Z2_0h <= d.D_path_0h + 1e-12)
    assert np.allclose(d.ratio_Dber_Z2_Dpath_0h, d.D_bern_Z2_0h / d.D_path_0h)
    assert np.all(d.D_path_h0 <= d.C_activity_h0 + 1e-12)
    assert np.all(d.D_path_0h <= d.C_activity_0h + 1e-12)
    assert int(d.nreal.sum()) == 5200

    b = pd.read_csv(BOOT)
    assert len(b) == len(d)
    assert np.allclose(b.tau_Q, d.tau_Q)
    assert np.allclose(b.g, d.g)

    print("Milestone 3 path-space audit")
    print(f"points={len(d)}; total production realizations={int(d.nreal.sum())}")
    print(f"exact-Z2 D_Ber/D_path point estimates = {d.ratio_Dber_Z2_Dpath_0h.min():.3f}--{d.ratio_Dber_Z2_Dpath_0h.max():.3f}")
    print(f"reverse D_path / finite-h activity ceiling = {d.ratio_Dpath0h_C0.min():.6f}--{d.ratio_Dpath0h_C0.max():.6f}")
    print(f"D_path / (Rmax^2/2) = {d.ratio_Dpath_S.min():.4f}--{d.ratio_Dpath_S.max():.4f}; mean={d.ratio_Dpath_S.mean():.4f}")
    print(f"wrong / rigorous reverse-KL lower-bound point estimate = {(d.wrong_on/d.wrong_lower_reverse_activity).min():.4f}--{(d.wrong_on/d.wrong_lower_reverse_activity).max():.4f}")
    print(f"all reactant-cap events = {int(d.clips.sum())}")
    print("bootstrap 95% intervals archived for D_path, exact-Z2 D_Ber, q, q_min, and ceiling ratio")

    dt = pd.read_csv(DTC)
    print("dt convergence:")
    for _, r in dt.sort_values("dt", ascending=False).iterrows():
        print(f"  dt={r['dt']:.3f}: D_h0={r['D_path_h0']:.6f}, p_on={r['p_on']:.4f}, p0={r['p_off']:.4f}, clips={int(r['clips'])}")


if __name__ == "__main__":
    main()
