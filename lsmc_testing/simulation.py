"""Risk-neutral path simulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def simulate_gbm_paths(
    *,
    s0: float,
    rate: float,
    sigma: float,
    maturity: float,
    steps_per_year: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = True,
    dtype: np.dtype = np.float64,
) -> NDArray[np.float64]:
    """Simulate GBM paths with shape ``(n_steps + 1, n_paths)``."""
    n_steps = int(round(maturity * steps_per_year))
    if n_steps < 1:
        raise ValueError("maturity * steps_per_year must produce at least one step")
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")

    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    drift = (rate - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    paths = np.empty((n_steps + 1, n_paths), dtype=dtype)
    paths[0] = s0
    for step in range(1, n_steps + 1):
        if antithetic:
            half = (n_paths + 1) // 2
            normals = rng.standard_normal(half)
            shocks = np.concatenate([normals, -normals])[:n_paths]
        else:
            shocks = rng.standard_normal(n_paths)
        paths[step] = paths[step - 1] * np.exp(drift + diffusion * shocks)
    return paths


def simulate_heston_spot_scaled_vol_of_vol_paths(
    *,
    s0: float,
    rate: float,
    v0: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
    maturity: float,
    steps_per_year: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = True,
    return_variance: bool = False,
    return_floor_hit_rate: bool = False,
    variance_scheme: str = "euler",
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """Simulate Heston paths with spot-scaled volatility-of-volatility.

    The model is discretized as:

    dS_t = r S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa(theta - v_t)dt + vol_of_vol S_t sqrt(v_t) dW2_t

    Variance is evolved with full truncation so negative simulated variance
    values are floored at zero before they enter drift and diffusion terms.
    Use ``variance_scheme="euler"`` for Euler-Maruyama or
    ``variance_scheme="milstein"`` for a Milstein variance update.
    """
    n_steps = int(round(maturity * steps_per_year))
    if n_steps < 1:
        raise ValueError("maturity * steps_per_year must produce at least one step")
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")
    if v0 < 0.0:
        raise ValueError("v0 must be non-negative")
    if theta < 0.0:
        raise ValueError("theta must be non-negative")
    if 2.0 * kappa * theta <= vol_of_vol * vol_of_vol:
        raise ValueError("Heston parameters must satisfy the Feller condition: 2 * kappa * theta > vol_of_vol ** 2")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be between -1 and 1")
    if variance_scheme not in {"euler", "milstein"}:
        raise ValueError('variance_scheme must be either "euler" or "milstein"')

    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)

    spot_paths = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    spot_paths[0] = s0
    variance_paths = np.empty((n_steps + 1, n_paths), dtype=np.float64) if return_variance else None
    if variance_paths is not None:
        variance_paths[0] = v0
    previous_variance = np.full(n_paths, v0, dtype=np.float64)
    floor_hit_count = 0

    for step in range(1, n_steps + 1):
        z_spot, z_variance = _correlated_normals(rng, n_paths, rho, antithetic)
        previous_spot = spot_paths[step - 1]
        positive_variance = np.maximum(previous_variance, 0.0)
        sqrt_variance = np.sqrt(positive_variance)

        spot_paths[step] = previous_spot * np.exp(
            (rate - 0.5 * positive_variance) * dt
            + sqrt_variance * sqrt_dt * z_spot
        )
        variance_drift = kappa * (theta - positive_variance) * dt
        variance_diffusion = vol_of_vol * previous_spot * sqrt_variance * sqrt_dt * z_variance
        variance_increment = variance_drift + variance_diffusion
        if variance_scheme == "milstein":
            milstein_correction = (
                0.25
                * vol_of_vol
                * vol_of_vol
                * previous_spot
                * previous_spot
                * dt
                * (z_variance * z_variance - 1.0)
            )
            variance_increment += np.where(positive_variance > 0.0, milstein_correction, 0.0)

        raw_next_variance = previous_variance + variance_increment
        floor_hit_count += int(np.count_nonzero(raw_next_variance < 0.0))
        previous_variance = np.maximum(raw_next_variance, 0.0)
        if variance_paths is not None:
            variance_paths[step] = previous_variance

    if return_variance and return_floor_hit_rate:
        assert variance_paths is not None
        return spot_paths, variance_paths, floor_hit_count / (n_steps * n_paths)
    if return_variance:
        assert variance_paths is not None
        return spot_paths, variance_paths
    return spot_paths


def _correlated_normals(
    rng: np.random.Generator,
    n_paths: int,
    rho: float,
    antithetic: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if antithetic:
        half = (n_paths + 1) // 2
        independent = rng.standard_normal((2, half))
        independent = np.concatenate([independent, -independent], axis=1)[:, :n_paths]
    else:
        independent = rng.standard_normal((2, n_paths))

    z_spot = independent[0]
    z_variance = rho * independent[0] + np.sqrt(1.0 - rho * rho) * independent[1]
    return z_spot, z_variance
