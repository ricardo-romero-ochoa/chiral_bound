#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'data'/'milestone2_rdme_validation.csv')
sz=pd.read_csv(ROOT/'data'/'milestone2_system_size.csv')
dt=pd.read_csv(ROOT/'data'/'milestone2_dt_convergence.csv')
sw=pd.read_csv(ROOT/'data'/'milestone2_bias_switch.csv')
fig,axs=plt.subplots(2,2,figsize=(8.2,6.4))

ax=axs[0,0]
for tq,grp in d[d.g>0].groupby('tau_Q'):
    ax.scatter(grp.Rmax,grp.Rin,label=fr'$\tau_Q={tq:g}$')
mx=max(d.Rmax.max(),d.Rin.max())*1.08
ax.plot([0,mx],[0,mx],ls='--',lw=1,label='identity')
ax.set(xlabel=r'paired-flux envelope $R_{\max}$',ylabel=r'RDME incoming probit $R_{\rm in}$',title='(a) Microscopic-to-freeze-out validation')
ax.legend(frameon=False,fontsize=8)

ax=axs[0,1]
xx=np.linspace(0,1.05,200)
ax.plot(xx,norm.sf(xx),ls='--',lw=1,label=r'$\Phi(-R_{\max})$')
for tq,grp in d[d.g>0].groupby('tau_Q'):
    ax.scatter(grp.Rmax,grp.wrong_in,label=fr'incoming, $\tau_Q={tq:g}$')
    ax.scatter(grp.Rmax,grp.wrong_out,marker='x',label=fr'outgoing, $\tau_Q={tq:g}$')
ax.set_yscale('log');ax.set(xlabel=r'$R_{\max}$',ylabel='wrong-sign fraction',title='(b) Instantaneous proxy is not a late-time bound')
ax.legend(frameon=False,fontsize=6,ncol=2)

ax=axs[1,0]
ax.plot(sz.omega,sz.Rin,'o-',label='RDME')
ax.plot(sz.omega,sz.Rmax,'s--',label=r'$R_{\max}\propto\sqrt{\Omega}$')
ax.set(xlabel=r'molecules/site scale $\Omega$',ylabel='incoming probit',title='(c) System-size scaling')
ax.legend(frameon=False,fontsize=8)

ax=axs[1,1]
ax.errorbar(dt.dt,dt.wrong_in,yerr=dt.sem_in,fmt='o-',capsize=3,label='time-step convergence')
on=sw[sw.switch_off_at_incoming==0].iloc[0];off=sw[sw.switch_off_at_incoming==1].iloc[0]
ax.axhline(on.wrong_out,ls='--',lw=1,label='outgoing: bias sustained')
ax.axhline(off.wrong_out,ls=':',lw=1,label='outgoing: bias switched off')
ax.set(xlabel=r'tau-leap $\Delta t$',ylabel='wrong-sign fraction',title='(d) Numerical and dynamical controls')
ax.legend(frameon=False,fontsize=7)
fig.tight_layout()
out=ROOT/'data'/'milestone2_rdme_validation.pdf'
fig.savefig(out,bbox_inches='tight')
print(out)
