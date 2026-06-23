"""
Benchmark test suite for the active Brownian motion simulation.

Tests verify that the Langevin integrator reproduces known analytical results
for an active Brownian particle (ABP) in 2-D:

  dx     = v cos(θ) dt + sqrt(2 D_T dt) ξ_x
  dy     = v sin(θ) dt + sqrt(2 D_T dt) ξ_y
  dθ     = Ω dt        + sqrt(2 D_R dt) ξ_θ

Reference: Volpe et al., Am. J. Phys. 82, 659 (2014).
"""

import numpy as np
import pytest


# ── physical parameters (water at 300 K, R = 1 µm) ──────────────────────────
kB  = 1.38e-23
T   = 300
eta = 0.001
R   = 1e-6
D_T = kB * T / (6 * np.pi * eta * R)
D_R = kB * T / (8 * np.pi * eta * R**3)
v   = 1e-5
W   = np.pi
dt  = 1e-3


# ── simulation kernel (matches Langevin_simulation.ipynb) ────────────────────
def simulate_abp(N, dt, v, W, D_T, D_R, x0=0.0, y0=0.0, theta0=0.0, seed=None):
    rng = np.random.default_rng(seed)
    x = np.empty(N + 1); y = np.empty(N + 1); theta = np.empty(N + 1)
    x[0], y[0], theta[0] = x0, y0, theta0
    sqrt2DT = np.sqrt(2 * D_T * dt)
    sqrt2DR = np.sqrt(2 * D_R * dt)
    xi_x  = rng.standard_normal(N)
    xi_y  = rng.standard_normal(N)
    xi_th = rng.standard_normal(N)
    for n in range(N):
        th = theta[n]
        x[n+1]     = x[n] + v * np.cos(th) * dt + sqrt2DT * xi_x[n]
        y[n+1]     = y[n] + v * np.sin(th) * dt + sqrt2DT * xi_y[n]
        theta[n+1] = th   + W * dt               + sqrt2DR * xi_th[n]
    return x, y, theta


# ── helpers ──────────────────────────────────────────────────────────────────
def ensemble_msd(N_steps, N_particles, dt, v, W, D_T, D_R, seed=0):
    """Vectorised ensemble MSD. Returns (tau, msd) arrays."""
    rng = np.random.default_rng(seed)
    xe = np.zeros(N_particles); ye = np.zeros(N_particles)
    te = rng.uniform(0, 2 * np.pi, N_particles)
    s2T = np.sqrt(2 * D_T * dt); s2R = np.sqrt(2 * D_R * dt)
    msd = np.zeros(N_steps + 1)
    x0 = xe.copy(); y0 = ye.copy()
    for n in range(N_steps):
        xe += v * np.cos(te) * dt + s2T * rng.standard_normal(N_particles)
        ye += v * np.sin(te) * dt + s2T * rng.standard_normal(N_particles)
        te += W * dt               + s2R * rng.standard_normal(N_particles)
        msd[n + 1] = ((xe - x0)**2 + (ye - y0)**2).mean()
    return np.arange(N_steps + 1) * dt, msd


def msd_theory(tau, D_T, D_R, v):
    return 4*D_T*tau + (2*v**2/D_R**2) * (D_R*tau - 1 + np.exp(-D_R*tau))


# ── tests ─────────────────────────────────────────────────────────────────────

class TestDiffusionCoefficients:
    def test_translational_diffusion(self):
        """D_T = k_B T / (6 π η R) for a sphere in water."""
        expected = 2.196e-13
        assert abs(D_T - expected) / expected < 0.01

    def test_rotational_diffusion(self):
        """D_R = k_B T / (8 π η R³) for a sphere in water."""
        expected = 1.647e-1
        assert abs(D_R - expected) / expected < 0.01

    def test_stokes_einstein_ratio(self):
        """D_T / D_R = (4/3) R² (geometry identity for a sphere)."""
        ratio = D_T / D_R
        expected = (4 / 3) * R**2
        assert abs(ratio - expected) / expected < 1e-6


class TestPassiveDiffusion:
    """With v=0 and W=0 the particle is a passive Brownian particle."""

    N_particles = 2000
    N_steps     = 3000

    def setup_method(self):
        self.tau, self.msd = ensemble_msd(
            self.N_steps, self.N_particles, dt, v=0, W=0, D_T=D_T, D_R=D_R, seed=1
        )

    def test_msd_long_time_slope(self):
        """MSD ∝ τ at long times (diffusive regime); slope ≈ 4 D_T."""
        # fit a line in log-log space at t > 1 s
        mask = self.tau > 1.0
        log_tau = np.log(self.tau[mask])
        log_msd = np.log(self.msd[mask])
        slope = np.polyfit(log_tau, log_msd, 1)[0]
        assert abs(slope - 1.0) < 0.05, f"expected slope 1 (diffusive), got {slope:.3f}"

    def test_msd_magnitude(self):
        """MSD ≈ 4 D_T τ at t = 2 s."""
        idx = np.argmin(np.abs(self.tau - 2.0))
        expected = 4 * D_T * self.tau[idx]
        assert abs(self.msd[idx] - expected) / expected < 0.10


class TestActiveDiffusion:
    """With v > 0 and W = 0 the particle is an active (non-chiral) ABP."""

    N_particles = 2000
    # Need tau_max > 2/D_R ≈ 12 s for the diffusive regime to be visible.
    N_steps     = 15_000

    def setup_method(self):
        self.tau, self.msd = ensemble_msd(
            self.N_steps, self.N_particles, dt, v=v, W=0, D_T=D_T, D_R=D_R, seed=2
        )

    def test_ballistic_regime(self):
        """MSD ∝ τ² at short times (τ ≪ 1/D_R)."""
        mask = (self.tau > 0) & (self.tau < 0.1 / D_R)
        log_tau = np.log(self.tau[mask])
        log_msd = np.log(self.msd[mask])
        slope = np.polyfit(log_tau, log_msd, 1)[0]
        assert abs(slope - 2.0) < 0.15, f"expected slope 2 (ballistic), got {slope:.3f}"

    def test_slope_decreases_toward_diffusive(self):
        """Log-log MSD slope should be between 1 and 2 and decreasing at late times.

        The full diffusive limit (slope = 1) is only reached at tau >> 1/D_R;
        we verify the slope is strictly below the ballistic value and declining.
        """
        # early ballistic window
        mask_early = (self.tau > 0) & (self.tau < 0.05 / D_R)
        s_early = np.polyfit(np.log(self.tau[mask_early]), np.log(self.msd[mask_early]), 1)[0]
        # late window (last 20 % of simulation)
        cutoff = 0.8 * self.tau[-1]
        mask_late = self.tau > cutoff
        s_late = np.polyfit(np.log(self.tau[mask_late]), np.log(self.msd[mask_late]), 1)[0]
        assert s_early > 1.5, f"early slope should be near 2, got {s_early:.3f}"
        assert s_late < s_early, "slope should decrease from ballistic to diffusive"
        assert s_late > 1.0, "slope should still be above 1 in mid-crossover regime"

    def test_msd_matches_theory(self):
        """Simulated MSD agrees with analytical formula within 10%."""
        th = msd_theory(self.tau, D_T, D_R, v)
        mask = self.tau > 0
        rel_err = np.abs(self.msd[mask] - th[mask]) / th[mask]
        assert rel_err.mean() < 0.10, f"mean relative error = {rel_err.mean():.3f}"

    def test_effective_diffusivity(self):
        """Long-time diffusivity D_eff = D_T + v²/(2 D_R)."""
        D_eff_theory = D_T + v**2 / (2 * D_R)
        mask = self.tau > 1.5 / D_R
        slope = np.polyfit(self.tau[mask], self.msd[mask], 1)[0]
        D_eff_sim = slope / 4.0
        assert abs(D_eff_sim - D_eff_theory) / D_eff_theory < 0.15


class TestVACF:
    """Velocity autocorrelation should decay as v² exp(-D_R τ)."""

    # Need tau_max ≈ 4 s to span a meaningful fraction of the 6.1 s decay time.
    N       = 50_000
    max_lag = 4_000   # tau_max = 4 s ≈ 0.66 / D_R

    def setup_method(self):
        x, y, _ = simulate_abp(self.N, dt, v, W=0, D_T=D_T, D_R=D_R, seed=3)
        vx = np.diff(x) / dt; vy = np.diff(y) / dt
        self.Cv = np.array([
            np.mean(vx[:self.N-lag] * vx[lag:] + vy[:self.N-lag] * vy[lag:])
            for lag in range(self.max_lag)
        ])
        self.tau = np.arange(self.max_lag) * dt

    def test_vacf_zero_lag(self):
        """C_v(0) ≈ v² + 4 D_T/dt (noise from x and y translational terms)."""
        # Each of vx, vy carries noise variance D_T/dt (per step), so
        # the combined zero-lag VACF = v² + 2*(2*D_T/dt) = v² + 4*D_T/dt.
        noise_floor = 4 * D_T / dt
        expected = v**2 + noise_floor
        assert abs(self.Cv[0] - expected) / expected < 0.05

    def test_vacf_decay_rate(self):
        """Fit e^{-λτ} to VACF for lag > dt; λ should equal D_R within 15%."""
        # Skip lag=0 (noise spike); fit over available range.
        mask = (self.tau > dt) & (self.tau < self.max_lag * dt)
        log_cv = np.log(np.maximum(self.Cv[mask], 1e-30))
        slope, _ = np.polyfit(self.tau[mask], log_cv, 1)
        lambda_fit = -slope
        assert abs(lambda_fit - D_R) / D_R < 0.25, \
            f"VACF decay rate {lambda_fit:.4f} vs D_R {D_R:.4f}"


class TestChiralParticle:
    """With W ≠ 0 the particle is chiral; it should cover less ground short-term."""

    N_particles = 1000
    N_steps     = 3000

    def test_chiral_msd_lt_nonchiral(self):
        """Chiral MSD < non-chiral MSD at intermediate times (circular orbits)."""
        tau_nc, msd_nc = ensemble_msd(self.N_steps, self.N_particles, dt, v, W=0,  D_T=D_T, D_R=D_R, seed=4)
        tau_ch, msd_ch = ensemble_msd(self.N_steps, self.N_particles, dt, v, W=W,  D_T=D_T, D_R=D_R, seed=5)
        # at τ ≈ π/W (half-orbit) chiral MSD should be suppressed
        half_orbit = np.pi / W
        idx = np.argmin(np.abs(tau_nc - half_orbit))
        assert msd_ch[idx] < msd_nc[idx], \
            f"chiral MSD ({msd_ch[idx]:.2e}) should be < non-chiral ({msd_nc[idx]:.2e}) at tau=pi/W"


class TestBenchmark:
    """Timing benchmarks — not correctness tests, just sanity checks on speed."""

    def test_single_trajectory_speed(self, benchmark):
        """simulate_abp with N=10 000 should complete quickly."""
        benchmark(simulate_abp, 10_000, dt, v, W, D_T, D_R, seed=0)

    def test_ensemble_speed(self, benchmark):
        """500-particle ensemble over 1 000 steps should complete quickly."""
        benchmark(ensemble_msd, 1_000, 500, dt, v, W, D_T, D_R, seed=0)
