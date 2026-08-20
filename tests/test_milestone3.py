import numpy as np
import pytest

from pvchiral import FrankRDMEParams, simulate_postfreeze_branches
from pvchiral.pathinfo import bernoulli_kl, symmetric_pved_activity_kl_ceiling


def test_postfreeze_branch_path_information_small():
    p = FrankRDMEParams(omega=30.0, alpha=0.5)
    out = simulate_postfreeze_branches(
        0.01, 25.0, params=p, nx=8, nreal=8, dt=0.02,
        a_i=-0.25, burn_time=0.2, seed=4,
    )
    assert out['event_on'].shape == (8,)
    assert out['event_off'].shape == (8,)
    assert np.all(out['kl_h0'] >= 0)
    assert np.all(out['kl_0h'] >= 0)
    assert np.all(out['b2_activity_h'] >= 0)
    assert np.all(out['b2_activity_0'] >= 0)
    assert out['clip_events_on'] >= 0

    Bmax = 2.0
    cap_h = symmetric_pved_activity_kl_ceiling(0.01, np.mean(out['b2_activity_h']), Bmax)
    cap_0 = symmetric_pved_activity_kl_ceiling(0.01, np.mean(out['b2_activity_0']), Bmax)
    assert np.mean(out['kl_h0']) <= cap_h * 1.000001 + 1e-12
    assert np.mean(out['kl_0h']) <= cap_0 * 1.000001 + 1e-12


def test_binary_data_processing_observed_not_expected_to_be_exact_small():
    # Only a smoke-level finite-sample check with smoothing; theorem tests are analytic.
    p = FrankRDMEParams(omega=20.0, alpha=0.5)
    out = simulate_postfreeze_branches(
        0.006, 16.0, params=p, nx=6, nreal=40, dt=0.025,
        a_i=-0.30, burn_time=0.0, seed=12,
    )
    pon = (out['event_on'].sum()+0.5)/(len(out['event_on'])+1.0)
    poff = (out['event_off'].sum()+0.5)/(len(out['event_off'])+1.0)
    d = bernoulli_kl(pon, poff)
    # Loose finite-sample smoke assertion: observed Bernoulli divergence is not
    # absurdly larger than the path budget.
    assert d <= np.mean(out['kl_h0']) + 0.5


def test_archived_exact_z2_columns_are_self_consistent():
    """Regression guard for the production summary's exact-Z2 columns."""
    from pathlib import Path
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    d = pd.read_csv(root / "data" / "milestone3_pathinfo.csv")
    expected = d["D_bern_Z2_0h"] / d["D_path_0h"]
    assert np.allclose(d["ratio_Dber_Z2_Dpath_0h"], expected, rtol=1e-13, atol=1e-15)
    assert np.all(d["D_bern_Z2_0h"] <= d["D_path_0h"] + 1e-12)
    assert int(d["nreal"].sum()) == 5200
