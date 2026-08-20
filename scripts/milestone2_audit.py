#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
VAL=ROOT/'data'/'milestone2_rdme_validation.csv'
SIZE=ROOT/'data'/'milestone2_system_size.csv'
DT=ROOT/'data'/'milestone2_dt_convergence.csv'
SW=ROOT/'data'/'milestone2_bias_switch.csv'


def z_sem(p, sem_p, z):
    return sem_p / norm.pdf(z)


def main():
    d=pd.read_csv(VAL)
    weak=d[d.g>0].copy()
    weak['R_se']=[z_sem(p,se,z) for p,se,z in zip(weak.wrong_in,weak.sem_in,weak.Rin)]
    x=weak.Rmax.to_numpy(); y=weak.Rin.to_numpy(); w=1/weak.R_se.to_numpy()**2
    slope=float(np.sum(w*x*y)/np.sum(w*x*x))
    slope_se=float(np.sqrt(1/np.sum(w*x*x)))
    rmse=float(np.sqrt(np.mean((y-x)**2)))
    print(f'weak-field incoming-freeze weighted slope through origin = {slope:.6f} +/- {slope_se:.6f}')
    print(f'RMSE(Rin-Rmax) = {rmse:.6f}')
    for tq,grp in weak.groupby('tau_Q'):
        xx=grp.Rmax.to_numpy(); yy=grp.Rin.to_numpy()
        s=float(xx@yy/(xx@xx)); r=float(np.sqrt(np.mean((yy-xx)**2)))
        print(f'tau_Q={tq:g}: slope0={s:.6f}, RMSE={r:.6f}')

    # Check that the late-time/outgoing probability can violate the instantaneous
    # Incoming-freeze Gaussian p_wrong ~= Phi(-Rmax) proxy (not a late-time theorem).
    d['p_proxy']=norm.sf(d.Rmax)
    d['violation_factor']=d.p_proxy/np.maximum(d.wrong_out,1e-15)
    test=d[(d.tau_Q==100)&(d.g==0.015)].iloc[0]
    print('representative sustained-bias case:')
    print(f"  Gaussian incoming proxy p_wrong >= {test.p_proxy:.6f}")
    print(f"  measured outgoing p_wrong = {test.wrong_out:.6f}")
    print(f"  proxy/measured = {test.violation_factor:.2f}x")

    sz=pd.read_csv(SIZE)
    print('system-size scaling ratios Rin/Rmax:', ', '.join(f'{v:.3f}' for v in sz.ratio_Rin_Rmax))
    print(f"mean ratio={sz.ratio_Rin_Rmax.mean():.4f}, max |ratio-1|={np.max(np.abs(sz.ratio_Rin_Rmax-1)):.4f}")
    dt=pd.read_csv(DT)
    print(f"dt convergence p_wrong range={dt.wrong_in.min():.4f}--{dt.wrong_in.max():.4f}; all clip_events={int(dt.clip_events.sum())}")
    sw=pd.read_csv(SW)
    on=sw[sw.switch_off_at_incoming==0].iloc[0]; off=sw[sw.switch_off_at_incoming==1].iloc[0]
    print(f"bias-on outgoing p_wrong={on.wrong_out:.4f}; bias-off-after-incoming p_wrong={off.wrong_out:.4f}")

if __name__=='__main__':
    main()
