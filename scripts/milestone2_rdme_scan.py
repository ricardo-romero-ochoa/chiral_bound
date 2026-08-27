#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from pvchiral.rdme import FrankRDMEParams, simulate_frank_quench

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'milestone2_rdme_scan.csv'
GS=[0.0,0.005,0.010,0.015,0.020,0.025]
TAUQS=[100.0,200.0]
FIELDS=['tau_Q','g','dt','nx','nreal','a_hat','xi_hat','block','Lambda_resp','Rmax_block',
        'wrong_in','sem_in','Robs_in','wrong_out','sem_out','Robs_out','eta_in','eta_out',
        'meanN_in','meanN_out','clip_events','seed','steady_init']

def run(tau_values, gs, reset=False, nx=96, nreal=96, dt=0.01):
    rows=[] if reset or not OUT.exists() else list(csv.DictReader(OUT.open()))
    done={(float(r['tau_Q']),float(r['g']),float(r['dt']),int(r['nx']),int(r['nreal'])) for r in rows}
    p=FrankRDMEParams(omega=100.0,s0=1.0,D=1.0,alpha=0.5)
    for ti,tq in enumerate(tau_values):
        for gi,g in enumerate(gs):
            key=(tq,g,dt,nx,nreal)
            if key in done: continue
            seed=260818 + int(round(tq))*1000 + int(round(g*1e6))
            o=simulate_frank_quench(g,tq,params=p,nx=nx,nreal=nreal,dt=dt,a_i=-0.20,seed=seed)
            si=o['snapshots']['incoming_freeze']; so=o['snapshots']['outgoing_freeze']
            row=dict(tau_Q=tq,g=g,dt=dt,nx=nx,nreal=nreal,a_hat=o['a_hat'],xi_hat=o['xi_hat'],
                     block=o['block'],Lambda_resp=o['Lambda_resp_freeze'],Rmax_block=o['Rmax_block'],
                     wrong_in=si['wrong_fraction'],sem_in=si['wrong_sem'],Robs_in=si['probit_R'],
                     wrong_out=so['wrong_fraction'],sem_out=so['wrong_sem'],Robs_out=so['probit_R'],
                     eta_in=si['eta_mean'],eta_out=so['eta_mean'],meanN_in=si['mean_total_count'],
                     meanN_out=so['mean_total_count'],clip_events=o['clip_events'],seed=seed,steady_init=1)
            rows.append({k:row[k] for k in FIELDS}); done.add(key)
            rows=sorted(rows,key=lambda r:(float(r['tau_Q']),float(r['g']),float(r['dt'])))
            OUT.parent.mkdir(exist_ok=True)
            with OUT.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
            print(f"tau={tq:g} g={g:.4f} Rmax={o['Rmax_block']:.3f} Rin={si['probit_R']:.3f} "
                  f"Rout={so['probit_R']:.3f} pin={si['wrong_fraction']:.3f} pout={so['wrong_fraction']:.3f}",flush=True)
    return rows

def summarize(rows):
    for tq in sorted({float(r['tau_Q']) for r in rows}):
        rr=[r for r in rows if float(r['tau_Q'])==tq and float(r['g'])>0]
        if len(rr)<3: continue
        x=np.array([float(r['Rmax_block']) for r in rr]); y=np.array([float(r['Robs_in']) for r in rr])
        slope0=float(x@y/(x@x)); slope,inter=np.polyfit(x,y,1)
        rmse=float(np.sqrt(np.mean((y-x)**2)))
        print(f"summary tau={tq:g}: Rin vs Rmax slope0={slope0:.3f}, slope={slope:.3f}, intercept={inter:.3f}, RMSE_to_identity={rmse:.3f}")

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tau',type=float,action='append');ap.add_argument('--g',type=float,action='append')
    ap.add_argument('--reset',action='store_true');ap.add_argument('--nx',type=int,default=96);ap.add_argument('--nreal',type=int,default=96);ap.add_argument('--dt',type=float,default=0.01)
    a=ap.parse_args(); rows=run(a.tau or TAUQS,a.g or GS,a.reset,a.nx,a.nreal,a.dt);summarize(rows)
