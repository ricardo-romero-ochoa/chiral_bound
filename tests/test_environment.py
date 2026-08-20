"""Finite-band environmental spectra and filtering."""
import numpy as np
import pytest
from pvchiral import (
    a_hat_spectral,
    fastest_decade_dominates,
    normalized_power_law_psd,
    spectral_slew,
)


def test_psd_normalization():
    w = np.geomspace(1e-6, 1e-2, 10000)
    S = normalized_power_law_psd(w, beta=2.0, variance=3.0)
    assert np.trapezoid(S, w) == pytest.approx(3.0, rel=1e-8)


def test_fastest_decade_condition_is_beta_less_than_three():
    assert fastest_decade_dominates(2.9)
    assert not fastest_decade_dominates(3.0)
    assert not fastest_decade_dominates(4.0)


def test_filter_reduces_slew_and_freezeout_distance():
    args = dict(sigma_a=2e-4, beta=2.0, omega_min=2e-7, omega_max=7e-5)
    raw = spectral_slew(**args, tau_filter=0.0)
    filt = spectral_slew(**args, tau_filter=1e7)
    assert filt["v_rms"] < raw["v_rms"]
    assert a_hat_spectral(**args, tau_filter=1e7) < a_hat_spectral(**args, tau_filter=0.0)
