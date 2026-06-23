# Active Brownian Motion — Langevin Simulation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hisaylama/Brownian-Simulation/blob/master/Langevin_simulation.ipynb)

A complete Python simulation of an **active Brownian particle (ABP)** — a self-propelled microswimmer — in a homogeneous 2-D fluid, implemented via the overdamped Langevin equations:

$$\dot{x} = v\cos\theta + \sqrt{2D_T}\,\xi_x(t)$$
$$\dot{y} = v\sin\theta + \sqrt{2D_T}\,\xi_y(t)$$
$$\dot{\theta} = \Omega + \sqrt{2D_R}\,\xi_\theta(t)$$

| Symbol | Meaning |
|---|---|
| $v$ | Self-propulsion speed |
| $\theta$ | Orientation angle |
| $\Omega$ | Angular velocity (chirality) |
| $D_T$ | Translational diffusion coefficient |
| $D_R$ | Rotational diffusion coefficient |
| $\xi$ | Independent Gaussian white-noise terms |

> **Reference:** Volpe et al., *Simulation of the active Brownian motion of microswimmers*, Am. J. Phys. **82**, 659 (2014).

---

## Repository contents

| File | Description |
|---|---|
| `Langevin_simulation.ipynb` | Main simulation notebook (run in Colab or Jupyter) |
| `test_brownian.py` | Pytest benchmark suite — 12 correctness tests + 2 benchmarks |
| `Borwnian_Simulation_active_swimmer.m` | Original MATLAB implementation |
| `Lotka_Voltera_Solution.ipynb` | Lotka–Volterra predator-prey dynamics + ML |

---

## Physical parameters

| Quantity | Symbol | Value | Notes |
|---|---|---|---|
| Temperature | $T$ | 300 K | Room temperature |
| Dynamic viscosity | $\eta$ | 0.001 Pa·s | Water at ~20 °C |
| Particle radius | $R$ | 1 µm | Typical microswimmer |
| Translational diffusion | $D_T$ | 2.20 × 10⁻¹³ m²/s | Stokes–Einstein |
| Rotational diffusion | $D_R$ | 0.165 rad²/s | Stokes–Einstein |
| Persistence time | $\tau_R = 1/D_R$ | ~6.1 s | Orientational memory |
| Self-propulsion speed | $v$ | 10 µm/s | Active swimmer |
| Angular velocity | $\Omega$ | π rad/s | One revolution per ~2 s |
| Time step | $dt$ | 1 ms | Euler–Maruyama |
| Steps per trajectory | $N$ | 10 000 | 10 s total |

---

## How to use the code

### Option 1 — Google Colab (no installation needed)

Click the badge at the top of this README or at the top of `Langevin_simulation.ipynb`.

### Option 2 — Local Jupyter

```bash
pip install numpy matplotlib jupyter
jupyter notebook Langevin_simulation.ipynb
```

### Option 3 — Run the simulation as a plain Python script

Copy the `simulate_abp` function from the notebook or `test_brownian.py`:

```python
import numpy as np
import matplotlib.pyplot as plt

kB = 1.38e-23; T = 300; eta = 0.001; R = 1e-6
D_T = kB * T / (6 * np.pi * eta * R)   # 2.20e-13 m²/s
D_R = kB * T / (8 * np.pi * eta * R**3) # 0.165 rad²/s
dt = 1e-3; N = 10_000; v = 1e-5; W = np.pi

def simulate_abp(N, dt, v, W, D_T, D_R, seed=None):
    rng = np.random.default_rng(seed)
    x = np.empty(N+1); y = np.empty(N+1); theta = np.empty(N+1)
    x[0] = y[0] = theta[0] = 0.0
    s2T = np.sqrt(2*D_T*dt); s2R = np.sqrt(2*D_R*dt)
    xi_x = rng.standard_normal(N)
    xi_y = rng.standard_normal(N)
    xi_th = rng.standard_normal(N)
    for n in range(N):
        th = theta[n]
        x[n+1]     = x[n] + v*np.cos(th)*dt + s2T*xi_x[n]
        y[n+1]     = y[n] + v*np.sin(th)*dt + s2T*xi_y[n]
        theta[n+1] = th   + W*dt             + s2R*xi_th[n]
    return x, y, theta

x, y, theta = simulate_abp(N, dt, v, W, D_T, D_R, seed=42)
plt.plot(x*1e6, y*1e6); plt.xlabel('x [µm]'); plt.ylabel('y [µm]'); plt.show()
```

### Run the test suite

```bash
pip install pytest pytest-benchmark
pytest test_brownian.py -v                  # correctness tests only
pytest test_brownian.py -v --benchmark-only # timing benchmarks
```

---

## Simulation results

### Single-particle trajectory (10 s, coloured by time)

![Single particle trajectory](single_trajectory.png)

The particle self-propels at speed $v = 10$ µm/s while rotational diffusion gradually randomises its heading. The colour scale shows elapsed time — early motion (purple) is nearly straight (ballistic), while later motion (yellow) is randomised.

---

### Velocity autocorrelation function (VACF)

![Velocity autocorrelation](vacf.png)

The VACF decays exponentially as $C_v(\tau) = v^2 e^{-D_R \tau}$, with decay time $\tau_R = 1/D_R \approx 6.1$ s. The simulation (blue) matches the analytical prediction (red dashed) closely.

---

### Mean-square displacement — ensemble of 500 particles

![Mean-square displacement](msd.png)

The MSD shows two distinct regimes separated at $\tau \sim 1/D_R$:

| Regime | Condition | Scaling | Physics |
|---|---|---|---|
| **Ballistic** | $\tau \ll 1/D_R$ | $\langle r^2\rangle \approx v^2\tau^2$ | Straight-line swimming |
| **Diffusive** | $\tau \gg 1/D_R$ | $\langle r^2\rangle \approx 4D_\text{eff}\tau$ | Random walk |

Effective diffusivity: $D_\text{eff} = D_T + \dfrac{v^2}{2D_R} \approx 3.04 \times 10^{-10}$ m²/s — about **1400× larger** than passive Brownian diffusion alone.

The simulation (blue) agrees with the analytical formula (red dashed):
$$\langle r^2(\tau)\rangle = 4D_T\tau + \frac{2v^2}{D_R^2}\left[D_R\tau - 1 + e^{-D_R\tau}\right]$$

---

### Effect of chirality ($\Omega = 0$ vs $\Omega \neq 0$)

![Chirality comparison](chirality_comparison.png)

Left to right:
- **Passive** ($v=0$): pure Brownian diffusion — compact random walk
- **Active, non-chiral** ($\Omega=0$): directed swimming with gradual randomisation
- **Chiral** ($\Omega = \pi$ rad/s): circular orbits at short times, diffusive at long times

---

## Test coverage

```
pytest test_brownian.py -v
```

| Test class | Tests | What is verified |
|---|---|---|
| `TestDiffusionCoefficients` | 3 | Stokes–Einstein $D_T$, $D_R$; ratio $D_T/D_R = \tfrac{4}{3}R^2$ |
| `TestPassiveDiffusion` | 2 | Linear MSD slope; correct magnitude $4D_T\tau$ |
| `TestActiveDiffusion` | 4 | Ballistic slope ~2; crossover to slower growth; theory match <10%; effective $D_\text{eff}$ |
| `TestVACF` | 2 | Zero-lag noise floor $v^2 + 4D_T/dt$; decay rate ≈ $D_R$ |
| `TestChiralParticle` | 1 | Chiral MSD suppressed vs non-chiral at half-orbit time |
| `TestBenchmark` | 2 | Timing benchmarks (need `--benchmark-only`) |

All 12 correctness tests pass in ~20 s on a standard laptop.

---

## Key physics

| Result | Value |
|---|---|
| $D_T$ (translational) | 2.20 × 10⁻¹³ m²/s |
| $D_R$ (rotational) | 0.165 rad²/s |
| Persistence time $1/D_R$ | 6.1 s |
| Effective swim diffusivity $v^2/(2D_R)$ | 3.04 × 10⁻¹⁰ m²/s |
| Enhancement over passive $D_\text{eff}/D_T$ | ~1400× |
