#!/usr/bin/env python3
"""Regenerate Milestone-2 RDME validation datasets (computationally expensive)."""
from __future__ import annotations
import csv
from pathlib import Path
from pvchiral.rdme import FrankRDMEParams, simulate_frank_quench

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
P=FrankRDMEParams(omega=100.0,s0=1.0,D=1.0,alpha=0.5)

def run_case(g,tq,nx,nreal,seed,burn,omega=100,dt=.01,off=False):
    p=FrankRDMEParams(omega=omega,s0=1.0,D=1.0,alpha=.5)
    o=simulate_frank_quench(g,tq,params=p,nx=nx,nreal=nreal,dt=dt,a_i=-.2,
                            seed=seed,burn_time=burn,
                            switch_off_bias_at_incoming_freeze=off)
    si=o['snapshots']['incoming_freeze']; so=o['snapshots']['outgoing_freeze']
    return o,si,so

def write(path,rows):
    path.parent.mkdir(exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(path)

def main():
    rows=[]
    for tq,nx,nreal,seed0 in [(100,56,48,950000),(200,48,40,960000)]:
        for g in [0,.005,.01,.015]:
            seed=seed0+int(g*1e6)
            o,si,so=run_case(g,tq,nx,nreal,seed,10)
            rows.append(dict(tau_Q=tq,g=g,nx=nx,nreal=nreal,dt=.01,burn_time=10,omega=100,
                a_hat=o['a_hat'],block=o['block'],Lambda_resp=o['Lambda_resp_freeze'],Rmax=o['Rmax_block'],
                wrong_in=si['wrong_fraction'],sem_in=si['wrong_sem'],Rin=si['probit_R'],
                wrong_out=so['wrong_fraction'],Rout=so['probit_R'],seed=seed))
    write(DATA/'milestone2_rdme_validation.csv',rows)

    rows=[]
    for om in [50,100,200]:
        seed=930000+om;o,si,so=run_case(.015,100,48,40,seed,15,omega=om)
        rows.append(dict(omega=om,tau_Q=100,g=.015,nx=48,nreal=40,dt=.01,burn_time=15,
            Rmax=o['Rmax_block'],wrong_in=si['wrong_fraction'],sem_in=si['wrong_sem'],Rin=si['probit_R'],
            ratio_Rin_Rmax=si['probit_R']/o['Rmax_block'],corr_length=si['corr_length'],clip_events=o['clip_events'],seed=seed))
    write(DATA/'milestone2_system_size.csv',rows)

    rows=[]
    for dt in [.02,.01,.005]:
        o,si,so=run_case(.015,100,40,40,940000,10,dt=dt)
        rows.append(dict(dt=dt,tau_Q=100,g=.015,omega=100,nx=40,nreal=40,burn_time=10,
            Rmax=o['Rmax_block'],wrong_in=si['wrong_fraction'],sem_in=si['wrong_sem'],Rin=si['probit_R'],
            ratio_Rin_Rmax=si['probit_R']/o['Rmax_block'],clip_events=o['clip_events'],seed=940000))
    write(DATA/'milestone2_dt_convergence.csv',rows)

    rows=[]
    for off in [False,True]:
        o,si,so=run_case(.015,100,56,48,970015,10,off=off)
        rows.append(dict(switch_off_at_incoming=int(off),tau_Q=100,g=.015,omega=100,nx=56,nreal=48,dt=.01,burn_time=10,
            Rmax=o['Rmax_block'],wrong_in=si['wrong_fraction'],Rin=si['probit_R'],wrong_out=so['wrong_fraction'],Rout=so['probit_R'],seed=970015))
    write(DATA/'milestone2_bias_switch.csv',rows)

if __name__=='__main__':
    main()
