"""Gas storage operation and Exhibit-2 style volume summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lsmc_testing.gas_storage_contract import GasStorageContract


@dataclass(frozen=True)
class VolumeBand:
    """Minimum and maximum attained volume across simulated paths."""

    minimum: NDArray[np.float64]
    maximum: NDArray[np.float64]


def intrinsic_schedule(
    prices: NDArray[np.float64],
    contract: GasStorageContract,
) -> NDArray[np.float64]:
    """Compute a deterministic least-cost schedule for a known price curve.

    This is a dynamic-programming intrinsic benchmark. It maximizes cashflow
    from injections/withdrawals over the supplied price curve subject to the
    Exhibit 1 volume constraints and terminal volume.
    """

    if prices.ndim != 1:
        raise ValueError("prices must be a one-dimensional daily curve")
    if len(prices) != contract.n_days + 1:
        raise ValueError("prices must contain n_days + 1 values")

    grid = np.asarray(contract.volume_grid, dtype=np.float64)
    n_grid = len(grid)
    n_days = contract.n_days
    value = np.full((n_days + 1, n_grid), -np.inf, dtype=np.float64)
    policy = np.zeros((n_days, n_grid), dtype=np.int32)
    terminal_index = _nearest_grid_index(grid, contract.end_volume)
    value[n_days, terminal_index] = 0.0

    for day in range(n_days - 1, -1, -1):
        days_remaining_after_step = n_days - day - 1
        for volume_index, volume in enumerate(grid):
            best_value = -np.inf
            best_next_index = volume_index
            for next_volume in contract.feasible_next_volumes(volume, days_remaining_after_step):
                next_index = _nearest_grid_index(grid, next_volume)
                cashflow = (volume - next_volume) * prices[day]
                candidate = cashflow + value[day + 1, next_index]
                if candidate > best_value:
                    best_value = candidate
                    best_next_index = next_index
            value[day, volume_index] = best_value
            policy[day, volume_index] = best_next_index

    volumes = np.empty(n_days + 1, dtype=np.float64)
    current_index = _nearest_grid_index(grid, contract.start_volume)
    volumes[0] = grid[current_index]
    for day in range(n_days):
        current_index = int(policy[day, current_index])
        volumes[day + 1] = grid[current_index]
    return volumes


def trigger_policy_volumes(
    price_paths: NDArray[np.float64],
    contract: GasStorageContract,
    *,
    low_quantile: float = 0.35,
    high_quantile: float = 0.65,
) -> NDArray[np.float64]:
    """Create pathwise volume schedules using simple spot-price trigger rules.

    This is a lightweight operational proxy for producing Exhibit-2 style
    min/max bands. Low prices inject, high prices withdraw, and middle prices
    hold, while terminal reachability is enforced every day.
    """

    if price_paths.ndim != 2:
        raise ValueError("price_paths must have shape (n_days + 1, n_paths)")
    if price_paths.shape[0] != contract.n_days + 1:
        raise ValueError("price_paths first dimension must equal n_days + 1")

    n_days, n_paths = contract.n_days, price_paths.shape[1]
    volumes = np.empty((n_days + 1, n_paths), dtype=np.float64)
    volumes[0] = contract.start_volume
    low_triggers = np.quantile(price_paths[:-1], low_quantile, axis=1)
    high_triggers = np.quantile(price_paths[:-1], high_quantile, axis=1)

    for day in range(n_days):
        current = volumes[day]
        proposed = current.copy()
        proposed = np.where(
            price_paths[day] <= low_triggers[day],
            current + contract.max_injection_per_day,
            proposed,
        )
        proposed = np.where(
            price_paths[day] >= high_triggers[day],
            current - contract.max_withdrawal_per_day,
            proposed,
        )
        days_remaining_after_step = n_days - day - 1
        lower = max(
            contract.min_volume,
            contract.end_volume - days_remaining_after_step * contract.max_injection_per_day,
        )
        upper = min(
            contract.max_volume,
            contract.end_volume + days_remaining_after_step * contract.max_withdrawal_per_day,
        )
        volumes[day + 1] = np.clip(proposed, lower, upper)

    return volumes


def volume_band(volumes: NDArray[np.float64]) -> VolumeBand:
    """Return minimum and maximum volume by day."""

    if volumes.ndim == 1:
        return VolumeBand(minimum=volumes.copy(), maximum=volumes.copy())
    return VolumeBand(minimum=volumes.min(axis=1), maximum=volumes.max(axis=1))


def _nearest_grid_index(grid: NDArray[np.float64], volume: float) -> int:
    return int(np.argmin(np.abs(grid - volume)))
