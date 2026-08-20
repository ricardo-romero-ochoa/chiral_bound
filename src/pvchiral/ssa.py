"""Exact time-inhomogeneous SSA checks for one spatial Frank-network cell.

The quench used by the RDME has propensities affine in time while the state is
fixed.  This permits exact sampling of the next event by integrating the linear
total hazard.  The module is intended as a verification path, not the main
production simulator.
"""
from __future__ import annotations

import math
import numpy as np

from .rdme import FrankRDMEParams, k3p_from_a, chemostat_A, kz_scales


def _control_affine(tau_Q: float, p: FrankRDMEParams):
    k30 = k3p_from_a(0.0, p)
    k3s = k3p_from_a(1.0/tau_Q, p) - k30
    A0 = chemostat_A(k30, p)
    A1 = chemostat_A(k30 + k3s, p)
    As = A1 - A0
    return k30, k3s, A0, As


def _channels(L, Dn, t: float, g: float, tau_Q: float, p: FrankRDMEParams):
    """Return affine channel rates at current t as (rate_now, slope, u, B2, action)."""
    nx=len(L); Om=p.omega; alpha=p.alpha
    k30,k3s,A0,As=_control_affine(tau_Q,p)
    k3=k30+k3s*t; A=A0+As*t
    out=[]
    # action tuples are (kind, site, target)
    for i in range(nx):
        l=int(L[i]); d=int(Dn[i])
        # R1 pair B=+1
        ep=math.exp(alpha*g); em=math.exp(-(1-alpha)*g)
        out.append((p.kp*ep*A*l, p.kp*ep*As*l, alpha*g, 1.0, ('bL',i,-1)))
        out.append((p.km*em*l*max(l-1,0)/Om, 0.0, -(1-alpha)*g, 1.0, ('dL',i,-1)))
        # R2 pair B=-1
        ep=math.exp(-alpha*g); em=math.exp((1-alpha)*g)
        out.append((p.kp*ep*A*d, p.kp*ep*As*d, -alpha*g, 1.0, ('bD',i,-1)))
        out.append((p.km*em*d*max(d-1,0)/Om, 0.0, (1-alpha)*g, 1.0, ('dD',i,-1)))
        # Mutual inhibition, B=0
        out.append((k3*l*d/Om, k3s*l*d/Om, 0.0, 0.0, ('ann',i,-1)))
        out.append((p.Q*Om, 0.0, 0.0, 0.0, ('revP',i,-1)))
        # Racemization pair B=-2; alpha=1/2 gives exponents -g,+g.
        out.append((p.kr*math.exp(-g)*l, 0.0, -g, 4.0, ('LD',i,-1)))
        out.append((p.kr*math.exp(g)*d, 0.0, g, 4.0, ('DL',i,-1)))
        # Diffusion: each direction has propensity D*n and no field response.
        if l:
            out.append((p.D*l,0.0,0.0,0.0,('diffL',i,(i-1)%nx)))
            out.append((p.D*l,0.0,0.0,0.0,('diffL',i,(i+1)%nx)))
        if d:
            out.append((p.D*d,0.0,0.0,0.0,('diffD',i,(i-1)%nx)))
            out.append((p.D*d,0.0,0.0,0.0,('diffD',i,(i+1)%nx)))
    return out


def _apply(L,Dn,act):
    kind,i,j=act
    if kind=='bL': L[i]+=1
    elif kind=='dL': L[i]-=1
    elif kind=='bD': Dn[i]+=1
    elif kind=='dD': Dn[i]-=1
    elif kind=='ann': L[i]-=1; Dn[i]-=1
    elif kind=='revP': L[i]+=1; Dn[i]+=1
    elif kind=='LD': L[i]-=1; Dn[i]+=1
    elif kind=='DL': Dn[i]-=1; L[i]+=1
    elif kind=='diffL': L[i]-=1; L[j]+=1
    elif kind=='diffD': Dn[i]-=1; Dn[j]+=1
    else: raise RuntimeError(kind)


def _hazard_integral(rate_now, slope, dt):
    return rate_now*dt + 0.5*slope*dt*dt


def simulate_exact_ssa_one_cell(
    g: float,
    tau_Q: float,
    *,
    params: FrankRDMEParams=FrankRDMEParams(),
    nx: int|None=None,
    nreal: int=128,
    seed: int=0,
):
    """Exact SSA from incoming to outgoing freeze-out from a symmetric Poisson state.

    Returns biased and zero-field final-sign samples plus realized log-likelihood
    ratios and integrated B^2 activities.  The initial law is identical and Z2
    symmetric in both ensembles.
    """
    if abs(params.alpha-0.5)>1e-12:
        raise ValueError('exact verification currently assumes alpha=1/2')
    sc=kz_scales(tau_Q,params)
    if nx is None: nx=max(1,int(np.rint(sc['xi_hat'])))
    t0=-sc['t_hat']; t1=sc['t_hat']
    rng=np.random.default_rng(seed)
    results={k:[] for k in ['event_g','event_0','loglr_g0','loglr_0g','klint_g0','klint_0g','Achi_g','Achi_0','events_g','events_0']}

    for rr in range(int(nreal)):
        # Independent symmetric initial states for each branch, drawn from the same law.
        # Sharing the exact state is unnecessary for KL/data processing, only the law must match.
        initL=rng.poisson(params.omega*params.s0/2.0,size=nx).astype(int)
        initD=rng.poisson(params.omega*params.s0/2.0,size=nx).astype(int)
        for field,label,lrdir in [(g,'g','g0'),(0.0,'0','0g')]:
            L=initL.copy(); Dn=initD.copy(); t=t0
            loglr=0.0; klint=0.0; Achi=0.0; nev=0
            while t < t1-1e-14:
                ch=_channels(L,Dn,t,field,tau_Q,params)
                rates=np.array([x[0] for x in ch],float)
                slopes=np.array([x[1] for x in ch],float)
                lam=float(rates.sum()); ms=float(slopes.sum())
                if lam<=0 and ms<=0:
                    dt=t1-t
                    break
                E=rng.exponential()
                if abs(ms)<1e-15:
                    dt=E/lam
                else:
                    disc=lam*lam+2.0*ms*E
                    dt=(-lam+math.sqrt(max(0.0,disc)))/ms
                if t+dt>=t1:
                    dt=t1-t
                    # integrate compensator and activity to endpoint, no event term
                    if lrdir=='g0':
                        # compare field path to zero
                        ch0=_channels(L,Dn,t,0.0,tau_Q,params)
                        diff0=np.array([a[0]-b[0] for a,b in zip(ch,ch0)])
                        diffs=np.array([a[1]-b[1] for a,b in zip(ch,ch0)])
                        loglr -= float(np.sum(diff0)*dt + 0.5*np.sum(diffs)*dt*dt)
                        uu=np.array([x[2] for x in ch],float)
                        phi=uu-1.0+np.exp(-uu)
                    else:
                        chg=_channels(L,Dn,t,g,tau_Q,params)
                        diff0=np.array([a[0]-b[0] for a,b in zip(ch,chg)])
                        diffs=np.array([a[1]-b[1] for a,b in zip(ch,chg)])
                        loglr -= float(np.sum(diff0)*dt + 0.5*np.sum(diffs)*dt*dt)
                        uu=np.array([x[2] for x in chg],float)
                        phi=-uu-1.0+np.exp(uu)
                    rr=np.array([x[0] for x in ch]); ss=np.array([x[1] for x in ch])
                    klint += float(np.sum(phi*(rr*dt+0.5*ss*dt*dt)))
                    b2r=np.array([x[3]*x[0] for x in ch]); b2s=np.array([x[3]*x[1] for x in ch])
                    Achi += float(b2r.sum()*dt+0.5*b2s.sum()*dt*dt)
                    t=t1; break

                # integrate to event
                if lrdir=='g0':
                    chref=_channels(L,Dn,t,0.0,tau_Q,params)
                    uu=np.array([x[2] for x in ch],float)
                    phi=uu-1.0+np.exp(-uu)
                else:
                    chref=_channels(L,Dn,t,g,tau_Q,params)
                    uu=np.array([x[2] for x in chref],float)
                    phi=-uu-1.0+np.exp(uu)
                diff0=np.array([a[0]-b[0] for a,b in zip(ch,chref)])
                diffs=np.array([a[1]-b[1] for a,b in zip(ch,chref)])
                loglr -= float(diff0.sum()*dt+0.5*diffs.sum()*dt*dt)
                rr=np.array([x[0] for x in ch]); ss=np.array([x[1] for x in ch])
                klint += float(np.sum(phi*(rr*dt+0.5*ss*dt*dt)))
                b2r=np.array([x[3]*x[0] for x in ch]); b2s=np.array([x[3]*x[1] for x in ch])
                Achi += float(b2r.sum()*dt+0.5*b2s.sum()*dt*dt)
                te=t+dt
                re=rates+slopes*dt
                total=float(re.sum())
                if total<=0: raise RuntimeError('nonpositive event rate')
                idx=int(np.searchsorted(np.cumsum(re),rng.random()*total,side='right'))
                idx=min(idx,len(ch)-1)
                # Event log rate ratio.
                if lrdir=='g0':
                    loglr += ch[idx][2]  # log w_g/w_0 = u
                else:
                    loglr -= _channels(L,Dn,te,g,tau_Q,params)[idx][2]  # log w_0/w_g=-u
                _apply(L,Dn,ch[idx][4])
                t=te; nev+=1
            m=int((L-Dn).sum())
            ev=1 if m>0 else 0 if m<0 else int(rng.integers(0,2))
            results[f'event_{label}'].append(ev)
            results[f'loglr_{lrdir}'].append(loglr)
            results[f'klint_{lrdir}'].append(klint)
            results[f'Achi_{label}'].append(Achi)
            results[f'events_{label}'].append(nev)
    for k,v in results.items(): results[k]=np.asarray(v)
    results.update({'g':float(g),'tau_Q':float(tau_Q),'nx':int(nx),'nreal':int(nreal),'t_hat':float(sc['t_hat'])})
    return results
