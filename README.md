[![DOI](https://zenodo.org/badge/1340286563.svg)](https://doi.org/10.5281/zenodo.22021975)

# chiral_bound 1.0.0

`chiral_bound` is a reproducibility repository for reaction-activity and path-information bounds in weakly biased chiral reaction networks. It contains the analytical helper functions, stochastic reaction-network implementations, archived numerical data, and regression tests used to validate the results.

## Core results implemented

### Paired-response bound

For a dimensionless perturbation `h`, define directional rate responses

```text
r_j+ = d ln k_j+ / dh
r_j- = d ln k_j- / dh
A_j  = r_j+ - r_j-
F_j  = (r_j+ + r_j-)/2
```

and, for each reversible reaction pair,

```text
a_j   = J_j+ + J_j-
rho_j = (J_j+ - J_j-)/a_j
Q_j   = A_j + 2 F_j rho_j
```

For an odd projected coordinate with chemical-Langevin noise `Theta`, total chiral-unit density `s`, and

```text
Lambda_resp = [sum_j a_j Q_j^2]/(2s),
```

the paired reaction channels obey

```text
|delta eps| <= |h| sqrt(s Theta Lambda_resp).
```

For parity-violating energy differences (PVED),

```text
g = Delta E_PV/(2 k_B T)
A_j = B_j = nu_j . chi
F_j = B_j (alpha_j - 1/2)
```

and symmetric barrier splitting (`alpha_j = 1/2`) gives `Lambda_resp = Lambda`.

### Finite-time fidelity bound

Let `P_h` and `P_0` be biased and unbiased continuous-time Markov path measures with the same initial distribution and support. For exponential rate perturbations

```text
w_rho^h = w_rho^0 exp(h r_rho),
```

define the reference response-weighted integrated activity

```text
A_r^(0) = E_0 integral_0^T dt sum_rho r_rho^2 w_rho^0.
```

The reverse path relative entropy satisfies

```text
D(P_0 || P_h) <= C_0
C_0 = (h^2/2) exp(|h| r_max) A_r^(0).
```

For a binary final sign event with exact zero-field `Z2` symmetry (`p_0 = 1/2`), data processing yields

```text
q_wrong >= [1 - sqrt(1 - exp(-2 C_0))]/2.
```

For symmetric PVED, the activity can be written in terms of chirality-changing reaction events, giving a necessary time-integrated turnover budget for a target final fidelity.

### Kibble-Zurek connection

For the critical left mode of a scalar supercritical pitchfork, the incoming-freeze signal scale is

```text
R_max = |g| sqrt(Lambda N_xi / a_hat).
```

When `N_xi` and `Lambda` vary weakly over the impulse interval,

```text
D_KZ ~= g^2 Lambda N_xi/(2 a_hat) = R_max^2/2.
```

`R_max` is treated as an incoming-freeze signal and path-information scale, not as a hard bound on the final domain probability.

## Archived validation data

The repository contains:

- a 162-point deterministic paired-response audit across two reaction-network topologies;
- spatial stochastic reaction-diffusion validation of the incoming-freeze scale;
- seven finite-time path-information conditions totaling 5200 realizations;
- realization-level arrays used for deterministic 20,000-resample bootstrap intervals;
- an exact time-inhomogeneous Gillespie consistency check;
- time-step and system-size consistency datasets.

The finite-time data distinguish simulated zero-field frequencies from the exact symmetry baseline. In `data/milestone3_pathinfo.csv`:

| column | definition | role |
|---|---|---|
| `D_bern_empirical_h0` | `d_Ber(p_on || p_off)` | empirical forward Bernoulli divergence |
| `D_bern_empirical_0h` | `d_Ber(p_off || p_on)` | empirical reverse Bernoulli divergence |
| `D_bern_Z2_0h` | `d_Ber(1/2 || p_on)` | exact-`Z2` quantity entering the finite-time theorem |
| `ratio_Dber_Z2_Dpath_0h` | `D_bern_Z2_0h / D_path_0h` | exact-`Z2` event/path information ratio |

The empirical zero-field branch is retained as a Monte Carlo symmetry diagnostic. The theorem uses `p_0 = 1/2` exactly.

## Repository layout

```text
src/pvchiral/
  bias.py             paired-response identities and bounds
  pathinfo.py         path-relative-entropy and final-fidelity bounds
  network.py          thermodynamically consistent reaction networks
  reduction.py        PVED activity/frenetic identities
  center_manifold.py  critical left-mode projection
  rdme.py             spatial stochastic Frank reaction-diffusion model
  ssa.py              exact time-inhomogeneous Gillespie implementation
  kz.py               Kibble-Zurek scales and incoming-freeze observables
  environment.py      finite-band environmental forcing and filtering
  kinetics.py         conditional kinetic ceilings
  ginzburg.py         Ginzburg consistency checks
  field1d.py          1D stochastic field checks
  field2d.py          compact 2D consistency checks
scripts/
  milestone1_audit.py
  milestone1_oligomer_grid.py
  milestone2_audit.py
  milestone2_generate.py
  milestone2_rdme_scan.py
  milestone3_audit.py
  milestone3_bootstrap.py
  milestone3_generate.py
  milestone3_ssa_verify.py
  run_kz_scan.py
  run_kz_scan_2d.py
  build_manifest.py
data/
  archived numerical tables, realization arrays, and validation graphics
tests/
  theorem, solver, RDME, SSA, and regression tests
```

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,figures]"
```

## Validation

Run the automated test suite:

```bash
make test
```

Run the archived numerical consistency checks:

```bash
make validate
```

The stochastic production scan is intentionally separate because it is computationally expensive:

```bash
make production
```

Rebuild the SHA-256 file manifest with:

```bash
make manifest
```

## Data regeneration

The finite-time production design is fixed in `scripts/milestone3_generate.py` to the archived sample counts and seeds. Realization-level final events, path-KL contributions, and chirality-weighted activities are written to `data/milestone3_realizations.npz`; bootstrap summaries can then be regenerated without rerunning the stochastic simulation.

## License

MIT. See `LICENSE`.
