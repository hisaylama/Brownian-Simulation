"""
brownian.py — Active Brownian motion simulation (standalone script)

Simulates a self-propelled microswimmer (active Brownian particle) in 2-D
using the overdamped Langevin equations:

    dx    = v cos(θ) dt + sqrt(2 D_T dt) ξ_x
    dy    = v sin(θ) dt + sqrt(2 D_T dt) ξ_y
    dθ    = Ω dt        + sqrt(2 D_R dt) ξ_θ

Usage
-----
Run directly:
    python brownian.py

Produces four figures:
    single_trajectory.png    — one particle trajectory coloured by time
    vacf.png                 — velocity autocorrelation vs theory
    msd.png                  — ensemble mean-square displacement vs theory
    chirality_comparison.png — passive / active / chiral side-by-side

Reference: Volpe et al., Am. J. Phys. 82, 659 (2014).
"""

import numpy as np
import matplotlib.pyplot as plt

# ── physical constants ────────────────────────────────────────────────────────
kB  = 1.38e-23   # Boltzmann constant  [J/K]
T   = 300        # temperature         [K]
eta = 0.001      # dynamic viscosity   [Pa·s]  (water at ~20 °C)
R   = 1e-6       # particle radius     [m]     (1 µm)

# ── diffusion coefficients (Stokes–Einstein) ──────────────────────────────────
D_T = kB * T / (6 * np.pi * eta * R)      # translational  [m²/s]
D_R = kB * T / (8 * np.pi * eta * R**3)   # rotational     [rad²/s]

# ── simulation parameters ─────────────────────────────────────────────────────
dt  = 1e-3       # time step            [s]
N   = 10_000     # steps per trajectory
v   = 1e-5       # self-propulsion speed [m/s]
W   = np.pi      # angular velocity      [rad/s]


def simulate_abp(N, dt, v, W, D_T, D_R, x0=0.0, y0=0.0, theta0=0.0, seed=None):
    """Integrate one active Brownian particle trajectory via Euler–Maruyama.

    Returns arrays x, y, theta of length N+1.
    """
    rng = np.random.default_rng(seed)
    x     = np.empty(N + 1)
    y     = np.empty(N + 1)
    theta = np.empty(N + 1)
    x[0], y[0], theta[0] = x0, y0, theta0
    s2T = np.sqrt(2 * D_T * dt)
    s2R = np.sqrt(2 * D_R * dt)
    xi_x  = rng.standard_normal(N)
    xi_y  = rng.standard_normal(N)
    xi_th = rng.standard_normal(N)
    for n in range(N):
        th         = theta[n]
        x[n+1]     = x[n] + v * np.cos(th) * dt + s2T * xi_x[n]
        y[n+1]     = y[n] + v * np.sin(th) * dt + s2T * xi_y[n]
        theta[n+1] = th   + W * dt               + s2R * xi_th[n]
    return x, y, theta


def ensemble_msd(N_steps, N_particles, dt, v, W, D_T, D_R, seed=0):
    """Vectorised ensemble mean-square displacement. Returns (tau, msd)."""
    rng = np.random.default_rng(seed)
    xe = np.zeros(N_particles)
    ye = np.zeros(N_particles)
    te = rng.uniform(0, 2 * np.pi, N_particles)
    s2T = np.sqrt(2 * D_T * dt)
    s2R = np.sqrt(2 * D_R * dt)
    msd = np.zeros(N_steps + 1)
    x0 = xe.copy(); y0 = ye.copy()
    for n in range(N_steps):
        xe += v * np.cos(te) * dt + s2T * rng.standard_normal(N_particles)
        ye += v * np.sin(te) * dt + s2T * rng.standard_normal(N_particles)
        te += W * dt               + s2R * rng.standard_normal(N_particles)
        msd[n + 1] = ((xe - x0)**2 + (ye - y0)**2).mean()
    return np.arange(N_steps + 1) * dt, msd


def msd_theory(tau, D_T, D_R, v):
    """Analytical MSD for an active Brownian particle."""
    return 4*D_T*tau + (2*v**2/D_R**2) * (D_R*tau - 1 + np.exp(-D_R*tau))


def plot_single_trajectory(N, dt, v, W, D_T, D_R, seed=42, save='single_trajectory.png'):
    x, y, theta = simulate_abp(N, dt, v, W, D_T, D_R, seed=seed)
    t = np.arange(N + 1) * dt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = axes[0].scatter(x*1e6, y*1e6, c=t, cmap='plasma', s=1, linewidths=0)
    axes[0].plot(x[0]*1e6,  y[0]*1e6,  'go', ms=8,  label='start', zorder=5)
    axes[0].plot(x[-1]*1e6, y[-1]*1e6, 'r*', ms=10, label='end',   zorder=5)
    plt.colorbar(sc, ax=axes[0], label='time [s]')
    axes[0].set_xlabel('x [µm]'); axes[0].set_ylabel('y [µm]')
    axes[0].set_title('Active Brownian particle trajectory')
    axes[0].set_aspect('equal'); axes[0].legend()
    axes[1].plot(t, np.unwrap(theta), color='steelblue', lw=0.8)
    axes[1].set_xlabel('time [s]'); axes[1].set_ylabel('θ [rad] (unwrapped)')
    axes[1].set_title('Orientation angle')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight'); plt.show()
    print(f'Saved {save}')
    return x, y, theta, t


def plot_vacf(x, y, N, dt, D_R, v, max_lag=2000, save='vacf.png'):
    vx = np.diff(x) / dt; vy = np.diff(y) / dt
    lags = np.arange(max_lag)
    Cv = np.array([np.mean(vx[:N-lag]*vx[lag:] + vy[:N-lag]*vy[lag:]) for lag in lags])
    tau = lags * dt
    plt.figure(figsize=(7, 4))
    plt.plot(tau, Cv, color='steelblue', lw=1, label='simulation')
    plt.plot(tau, v**2*np.exp(-D_R*tau), color='tomato', lw=1.5, ls='--',
             label=r'$v^2\,e^{-D_R\tau}$')
    plt.xlabel(r'lag $\tau$ [s]'); plt.ylabel(r'$C_v(\tau)$ [m²/s²]')
    plt.title('Velocity autocorrelation function')
    plt.legend(); plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight'); plt.show()
    print(f'Saved {save}')


def plot_msd(N_particles, N_steps, dt, v, W, D_T, D_R, seed=0, save='msd.png'):
    tau, msd = ensemble_msd(N_steps, N_particles, dt, v, W, D_T, D_R, seed=seed)
    D_eff = D_T + v**2 / (2 * D_R)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(tau[1:], msd[1:]*1e12, color='steelblue', lw=1.5, label='simulation')
    ax.loglog(tau[1:], msd_theory(tau, D_T, D_R, v)[1:]*1e12, color='tomato', lw=2, ls='--', label='theory')
    ax.loglog(tau[1:], v**2*tau[1:]**2*1e12, color='green', lw=1, ls=':', label=r'ballistic $\propto\tau^2$')
    ax.loglog(tau[1:], 4*D_eff*tau[1:]*1e12, color='orange', lw=1, ls=':', label=r'diffusive $\propto\tau$')
    ax.axvline(1/D_R, color='grey', ls='--', lw=0.8, label=r'$\tau=1/D_R$')
    ax.set_xlabel(r'lag $\tau$ [s]'); ax.set_ylabel(r'MSD [µm²]')
    ax.set_title(f'Mean-square displacement  ({N_particles} particles)')
    ax.legend(fontsize=9); plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight'); plt.show()
    print(f'Saved {save}')


def plot_chirality(N, dt, v, W, D_T, D_R, seed=7, save='chirality_comparison.png'):
    cases = [
        dict(label='passive (v=0, Ω=0)',       v=0, W=0),
        dict(label='active, no chirality',       v=v, W=0),
        dict(label=f'chiral (Ω={W:.2f} rad/s)', v=v, W=W),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, case in zip(axes, cases):
        xc, yc, _ = simulate_abp(N, dt, case['v'], case['W'], D_T, D_R, seed=seed)
        sc = ax.scatter(xc*1e6, yc*1e6, c=np.arange(N+1)*dt, cmap='viridis', s=1, linewidths=0)
        ax.plot(xc[0]*1e6, yc[0]*1e6, 'go', ms=7, zorder=5)
        ax.plot(xc[-1]*1e6, yc[-1]*1e6, 'r*', ms=9, zorder=5)
        plt.colorbar(sc, ax=ax, label='time [s]')
        ax.set_xlabel('x [µm]'); ax.set_ylabel('y [µm]')
        ax.set_title(case['label']); ax.set_aspect('equal')
    plt.suptitle('Trajectories: passive vs active vs chiral', fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight'); plt.show()
    print(f'Saved {save}')


if __name__ == '__main__':
    print(f'D_T = {D_T:.3e} m²/s')
    print(f'D_R = {D_R:.3e} rad²/s')
    print(f'Persistence time 1/D_R = {1/D_R:.2f} s')
    print(f'Effective diffusivity D_eff = {D_T + v**2/(2*D_R):.3e} m²/s')
    print()
    x, y, theta, t = plot_single_trajectory(N, dt, v, W, D_T, D_R)
    plot_vacf(x, y, N, dt, D_R, v)
    plot_msd(N_particles=500, N_steps=5_000, dt=dt, v=v, W=W, D_T=D_T, D_R=D_R)
    plot_chirality(N, dt, v, W, D_T, D_R)
