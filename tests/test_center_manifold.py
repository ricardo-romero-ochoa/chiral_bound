"""Critical-mode projection survives stable-mode elimination."""
import numpy as np
import pytest
from pvchiral import critical_field, critical_modes, projected_field_with_stable_shift


def test_left_zero_mode_annihilates_stable_shift():
    J = np.array([[0.0, 2.0, -1.0], [0.0, -2.0, 0.3], [0.0, 0.0, -4.0]])
    v, u = critical_modes(J)
    assert np.linalg.norm(J @ v) < 1e-10
    assert np.linalg.norm(u @ J) < 1e-10
    q = np.array([0.7, -0.2, 0.5])
    h_g = np.array([1.2, -3.0, 0.8])
    assert projected_field_with_stable_shift(J, q, h_g, left_mode=u) == pytest.approx(
        critical_field(J, q, left_mode=u), rel=1e-12, abs=1e-12)


def test_noncritical_observable_can_receive_stable_contribution():
    J = np.diag([0.0, -2.0])
    q = np.array([1.0, 0.0])
    h = np.array([0.0, 3.0])
    observable = np.array([1.0, 1.0])
    assert observable @ (q + J @ h) != pytest.approx(observable @ q)
