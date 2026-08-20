import numpy as np
from pvchiral.ssa import simulate_exact_ssa_one_cell
from pvchiral import FrankRDMEParams


def test_exact_ssa_smoke_and_nonnegative_kl_mean():
    p=FrankRDMEParams(omega=10.0,alpha=0.5,D=0.2)
    o=simulate_exact_ssa_one_cell(0.01,9.0,params=p,nx=2,nreal=8,seed=3)
    assert len(o['event_g'])==8
    assert np.mean(o['loglr_g0']) > -0.2
    assert np.mean(o['loglr_0g']) > -0.2
    assert np.all(o['Achi_g']>=0)
    assert np.all(o['Achi_0']>=0)
