"""Conditional, dimensionally consistent kinetic ceiling on Lambda."""
import pytest
from pvchiral import lambda_kinetic_ceiling


def test_kinetic_ceiling_formula():
    assert lambda_kinetic_ceiling(k_max=3.0, z=4.0, B_max=2.0) == pytest.approx(24.0)


def test_ceiling_requires_explicit_nonnegative_constraints():
    with pytest.raises(ValueError):
        lambda_kinetic_ceiling(-1.0, 2.0)
