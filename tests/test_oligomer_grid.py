"""Second-topology check that near-saturation of the paired bound is a property
of the pairing step, not of the Frank network."""
import csv
import pathlib
import numpy as np
import pytest

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "milestone1_oligomer_grid.csv"
pytestmark = pytest.mark.skipif(
    not DATA.exists(), reason="run scripts/milestone1_oligomer_grid.py first")


def _rows():
    return list(csv.DictReader(DATA.open()))


def test_grid_is_populated():
    assert len(_rows()) == 81


def test_no_bound_violations():
    for r in _rows():
        assert float(r["paired_saturation"]) <= 1.0 + 1e-9
        assert float(r["old_saturation"]) <= 1.0 + 1e-9


def test_paired_bound_is_near_saturated_on_second_topology():
    sat = np.array([float(r["paired_saturation"]) for r in _rows()])
    assert sat.min() > 0.97


def test_symmetric_split_saturates_exactly():
    """alpha = 1/2 gives Q_j = B_j = P_j, so Cauchy-Schwarz is an equality."""
    sat = [float(r["paired_saturation"]) for r in _rows() if float(r["alpha"]) == 0.5]
    assert sat and min(sat) == pytest.approx(1.0, abs=1e-6)


def test_pairing_improves_on_the_unpaired_envelope():
    rows = _rows()
    old = np.array([float(r["old_saturation"]) for r in rows])
    new = np.array([float(r["paired_saturation"]) for r in rows])
    assert np.all(new > old)
    assert old.max() < 0.60
