#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'data'/'milestone3_pathinfo.csv')
b=pd.read_csv(ROOT/'data'/'milestone3_bootstrap.csv')
fig,axs=plt.subplots(1,3,figsize=(10.8,3.25))

ax=axs[0]
x=d.D_path_0h.to_numpy(); y=d.D_bern_Z2_0h.to_numpy()
xerr=np.vstack([x-b.D_path_0h_lo.to_numpy(),b.D_path_0h_hi.to_numpy()-x])
yerr=np.vstack([y-b.D_bern_Z2_0h_lo.to_numpy(),b.D_bern_Z2_0h_hi.to_numpy()-y])
ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='o',capsize=2)
m=max(x.max(),y.max())*1.05
ax.plot([0,m],[0,m],'--',linewidth=1)
ax.set_xlabel(r'path KL $D(P_0\Vert P_g)$')
ax.set_ylabel(r'exact-$Z_2$ binary KL $d(1/2\Vert p_g)$')
ax.set_title('(a) Data processing')

ax=axs[1]
ax.scatter(d.S_gauss,d.D_path_h0,marker='o')
m=max(d.S_gauss.max(),d.D_path_h0.max())*1.05
ax.plot([0,m],[0,m],'--',linewidth=1)
ax.set_xlabel(r'$R_{\max}^2/2$')
ax.set_ylabel(r'path KL $D(P_g\Vert P_0)$')
ax.set_title('(b) KZ information action')

ax=axs[2]
x=d.wrong_lower_reverse_activity.to_numpy(); y=d.wrong_on.to_numpy()
xerr=np.vstack([x-b.qmin_lo.to_numpy(),b.qmin_hi.to_numpy()-x])
yerr=np.vstack([y-b.q_lo.to_numpy(),b.q_hi.to_numpy()-y])
ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='o',capsize=2)
m=max(y.max(),x.max())*1.08
ax.plot([0,m],[0,m],'--',linewidth=1)
ax.set_xlabel('rigorous lower bound on wrong sign')
ax.set_ylabel('measured wrong-sign probability')
ax.set_title('(c) Final-fidelity bound')

fig.tight_layout()
out=ROOT/'data'/'milestone3_pathspace_validation.pdf'
fig.savefig(out,bbox_inches='tight')
fig.savefig(ROOT/'data'/'milestone3_pathspace_validation.png',dpi=220,bbox_inches='tight')
print(out)
