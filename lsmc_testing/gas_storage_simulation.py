"""Daily simulation for the Boogert-de Jong gas spot process."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


MuFunction = Callable[[int], float]


@dataclass(frozen=True)
class GasPriceSimulationSpec:
    """Simulation inputs for Eq. 22 of Boogert and de Jong."""

    n_paths: int = 10_000
    n_days: int = 365
    initial_price: float = 20.0
    kappa: float = 0.05
    sigma: float = 0.0945
    seed: int = 20261002
    antithetic: bool = True


def constant_mu(level: float) -> MuFunction:
    """Return a constant long-term log-price level."""

    log_level = float(np.log(level))
    return lambda _day: log_level


def simulate_schwartz_spot_paths(
    spec: GasPriceSimulationSpec,
    *,
    mu: MuFunction,
) -> NDArray[np.float64]:
    """Simulate daily spot paths for dS/S = kappa(mu(t)-ln S)dt + sigma dW.

    The simulation is performed on X=ln(S). Applying Ito's lemma gives
    dX = [kappa(mu(t)-X) - 0.5 sigma^2]dt + sigma dW. The paper uses daily
    parameters, so one time step is one day and dt=1.
    """

    if spec.n_paths < 1:
        raise ValueError("n_paths must be positive")
    if spec.n_days < 1:
        raise ValueError("n_days must be positive")
    if spec.initial_price <= 0.0:
        raise ValueError("initial_price must be positive")

    rng = np.random.default_rng(spec.seed)
    log_paths = np.empty((spec.n_days + 1, spec.n_paths), dtype=np.float64)
    log_paths[0] = np.log(spec.initial_price)

    for day in range(spec.n_days):
        if spec.antithetic:
            half = (spec.n_paths + 1) // 2
            normals = rng.standard_normal(half)
            shocks = np.concatenate([normals, -normals])[: spec.n_paths]
        else:
            shocks = rng.standard_normal(spec.n_paths)

        previous_log_price = log_paths[day]
        drift = spec.kappa * (mu(day) - previous_log_price) - 0.5 * spec.sigma * spec.sigma
        log_paths[day + 1] = previous_log_price + drift + spec.sigma * shocks

    return np.exp(log_paths)


def path_summary(paths: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """Return min, mean, and max spot summaries by day."""

    return {
        "min": paths.min(axis=1),
        "mean": paths.mean(axis=1),
        "max": paths.max(axis=1),
    }
