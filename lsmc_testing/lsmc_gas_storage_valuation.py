"""LSMC gas storage valuation with OLS continuation values.

This module implements the continuation-value strategy described by Boogert
and de Jong for the Exhibit 1 gas storage contract and produces an Exhibit 6
style table. The original paper used the Dutch TTF forward curve observed on
2005-06-10; that curve is not included in this repository, so the default run
uses a documented seasonal proxy with the same broad July/February range.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numpy.typing import NDArray

from lsmc_testing.excel import ChartSpec, Sheet, write_workbook
from lsmc_testing.gas_storage_contract import GasStorageContract, exhibit1_contract
from lsmc_testing.gas_storage_simulation import GasPriceSimulationSpec, simulate_schwartz_spot_paths
from lsmc_testing.gas_storage_valuation import intrinsic_schedule


DEFAULT_OUTPUT_PATH = Path("output") / "gas_storage_exhibit6_lsmc.xlsx"
VOLUME_PLOT_PATH_COUNTS: tuple[int, ...] = (50,)
VOLUME_PLOT_BASIS_LABELS: tuple[str, ...] = ("P3",)
GAS_KAPPA = 0.05


@dataclass(frozen=True)
class BasisSpec:
    """Regression basis configuration for continuation values."""

    family: str
    count: int
    label: str


@dataclass(frozen=True)
class ValuationResult:
    """One LSMC valuation result."""

    value: float
    runtime_seconds: float
    volume_paths: NDArray[np.float64] | None = None
    path_values: NDArray[np.float64] | None = None
    daily_cashflows: NDArray[np.float64] | None = None
    accumulated_cashflows: NDArray[np.float64] | None = None
    continuation_values: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class Exhibit6Result:
    """Aggregated LSMC result for one basis/path-count pair."""

    basis: str
    path_count: int
    mean: float
    std_error: float
    average_runtime_seconds: float
    run_values: tuple[float, ...]
    run_std_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class VolumePlotData:
    """Pathwise storage volumes to chart for one basis/path-count pair."""

    basis: str
    path_count: int
    volumes: NDArray[np.float64]


@dataclass(frozen=True)
class PricePlotData:
    """Pathwise gas prices to chart for one path-count sample."""

    path_count: int
    prices: NDArray[np.float64]


@dataclass(frozen=True)
class StorageValueDistributionData:
    """Pathwise storage values for one valuation run."""

    basis: BasisSpec
    path_count: int
    values: NDArray[np.float64]
    intrinsic_value: float


@dataclass(frozen=True)
class ContinuationValueData:
    """Pathwise P3 cashflow and continuation diagnostics."""

    basis: BasisSpec
    path_count: int
    prices: NDArray[np.float64]
    volumes: NDArray[np.float64]
    daily_cashflows: NDArray[np.float64]
    accumulated_cashflows: NDArray[np.float64]
    continuation_values: NDArray[np.float64]


EXHIBIT6_BASIS_SPECS: tuple[BasisSpec, ...] = (
    BasisSpec("power", 4, "P3"),
    BasisSpec("power", 5, "P4"),
    BasisSpec("power", 6, "P5"),
    BasisSpec("power", 7, "P6"),
    BasisSpec("bspline", 5, "B5"),
    BasisSpec("bspline", 6, "B6"),
)

EXHIBIT6_PATH_COUNTS: tuple[int, ...] = (50, 500, 5000)
B6_CONVERGENCE_PATH_COUNTS: tuple[int, ...] = tuple(50 * (2**index) for index in range(9))
B6_CONVERGENCE_BASIS = BasisSpec("bspline", 6, "B6")

PAPER_EXHIBIT6: dict[tuple[str, int], tuple[float, float, float]] = {
    ("P3", 50): (5_469_006.0, 67_536.0, 1.0),
    ("P3", 500): (5_496_365.0, 22_884.0, 3.8),
    ("P3", 5000): (5_502_115.0, 5_341.0, 40.9),
    ("P4", 50): (5_449_743.0, 68_298.0, 1.0),
    ("P4", 500): (5_433_701.0, 24_246.0, 3.8),
    ("P4", 5000): (5_436_252.0, 3_368.0, 41.2),
    ("P5", 50): (5_463_046.0, 70_319.0, 1.1),
    ("P5", 500): (5_414_859.0, 24_213.0, 3.8),
    ("P5", 5000): (5_412_658.0, 4_294.0, 41.4),
    ("P6", 50): (5_473_720.0, 69_226.0, 1.1),
    ("P6", 500): (5_401_761.0, 22_241.0, 3.9),
    ("P6", 5000): (5_397_023.0, 4_171.0, 41.7),
    ("B5", 50): (5_466_878.0, 71_008.0, 3.4),
    ("B5", 500): (5_414_515.0, 24_477.0, 5.7),
    ("B5", 5000): (5_413_977.0, 4_295.0, 66.1),
    ("B6", 50): (5_474_144.0, 69_125.0, 7.0),
    ("B6", 500): (5_403_416.0, 21_749.0, 5.9),
    ("B6", 5000): (5_398_939.0, 4_462.0, 68.7),
}


def seasonal_ttf_proxy_curve(contract: GasStorageContract) -> NDArray[np.float64]:
    """Return a daily seasonal forward proxy for the 2005-06-10 TTF curve."""

    dates = contract.dates
    values = np.empty(len(dates), dtype=np.float64)
    low, high = 14.88, 25.44
    midpoint = 0.5 * (low + high)
    amplitude = 0.5 * (high - low)
    peak_day = date(2006, 2, 1).toordinal()
    for index, current_date in enumerate(dates):
        phase = 2.0 * np.pi * (current_date.toordinal() - peak_day) / 365.0
        values[index] = midpoint + amplitude * np.cos(phase)
    return np.maximum(values, 0.01)


def simulate_exhibit_price_paths(
    *,
    contract: GasStorageContract,
    n_paths: int,
    sigma: float,
    seed: int,
) -> NDArray[np.float64]:
    """Simulate daily spot paths for the Exhibit 6 high-volatility case."""

    forward_curve = seasonal_ttf_proxy_curve(contract)
    calibrated_mu = calibrate_mu_to_forward_curve(
        forward_curve,
        kappa=GAS_KAPPA,
        sigma=sigma,
    )
    spec = GasPriceSimulationSpec(
        n_paths=n_paths,
        n_days=contract.n_days,
        initial_price=float(forward_curve[0]),
        kappa=GAS_KAPPA,
        sigma=sigma,
        seed=seed,
        antithetic=True,
    )
    return simulate_schwartz_spot_paths(spec, mu=lambda day: float(calibrated_mu[day]))


def calibrate_mu_to_forward_curve(
    forward_curve: NDArray[np.float64],
    *,
    kappa: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Calibrate daily mean-reversion levels so E[S_t] matches the forward curve."""

    if forward_curve.ndim != 1:
        raise ValueError("forward_curve must be one-dimensional")
    if len(forward_curve) < 2:
        raise ValueError("forward_curve must contain at least two dates")
    if np.any(forward_curve <= 0.0):
        raise ValueError("forward_curve values must be positive")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    mu = np.empty(len(forward_curve) - 1, dtype=np.float64)
    log_forward = np.log(forward_curve)
    mean_log_spot = float(log_forward[0])
    variance_log_spot = 0.0
    for day in range(len(mu)):
        next_variance = (1.0 - kappa) ** 2 * variance_log_spot + sigma * sigma
        target_next_mean = float(log_forward[day + 1]) - 0.5 * next_variance
        mu[day] = (target_next_mean - (1.0 - kappa) * mean_log_spot + 0.5 * sigma * sigma) / kappa
        mean_log_spot = target_next_mean
        variance_log_spot = next_variance
    return mu


def intrinsic_contract_value(contract: GasStorageContract) -> float:
    """Return the deterministic intrinsic contract value for the proxy forward curve."""

    prices = seasonal_ttf_proxy_curve(contract)
    volumes = intrinsic_schedule(prices, contract)
    return float(np.sum((volumes[:-1] - volumes[1:]) * prices[:-1]))


def value_storage_lsmc(
    price_paths: NDArray[np.float64],
    contract: GasStorageContract,
    basis: BasisSpec,
    transitions: list[tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.bool_]]] | None = None,
    *,
    return_volume_paths: bool = False,
    return_path_diagnostics: bool = False,
    valuation_price_paths: NDArray[np.float64] | None = None,
) -> ValuationResult:
    """Value storage by backward OLS regression of continuation values.

    When valuation_price_paths is supplied, regressions are fitted on price_paths
    and the resulting policy is evaluated on the independent valuation paths.
    """

    start_time = time.perf_counter()
    _validate_paths(price_paths, contract)
    if valuation_price_paths is not None:
        _validate_paths(valuation_price_paths, contract)

    prices = price_paths[:-1]
    grid = np.asarray(contract.volume_grid, dtype=np.float64)
    n_days = contract.n_days
    n_grid = len(grid)
    n_paths = price_paths.shape[1]
    terminal_index = _nearest_grid_index(grid, contract.end_volume)
    if transitions is None:
        transitions = _transition_candidates(contract, grid)
    policy = None
    if return_volume_paths or return_path_diagnostics:
        if n_grid > np.iinfo(np.int16).max:
            raise ValueError("volume grid is too large to store path policy as int16")
        policy = np.zeros((n_days, n_grid, n_paths), dtype=np.int16)
    continuation_policy = np.zeros((n_days, n_grid, n_paths), dtype=np.float64) if return_path_diagnostics else None
    coefficients_by_day: list[NDArray[np.float64] | None] = [None] * n_days
    basis_scales = np.empty(n_days, dtype=np.float64)

    future_values = np.full((n_grid, n_paths), -1.0e12, dtype=np.float64)
    future_values[terminal_index] = 0.0

    for day in range(n_days - 1, -1, -1):
        basis_scales[day] = float(np.mean(prices[day]))
        design = build_basis_matrix(prices[day], basis, price_scale=basis_scales[day])
        coefficients = np.linalg.pinv(design.T @ design) @ design.T @ future_values.T
        coefficients_by_day[day] = coefficients
        continuation_by_next_volume = (design @ coefficients).T
        next_indices, volume_deltas, valid = transitions[day]
        current_values = np.full((n_grid, n_paths), -1.0e12, dtype=np.float64)
        if policy is not None:
            selected_next_indices = np.zeros((n_grid, n_paths), dtype=np.int16)
        if continuation_policy is not None:
            selected_continuation = np.zeros((n_grid, n_paths), dtype=np.float64)

        for candidate_index in range(next_indices.shape[1]):
            valid_rows = valid[:, candidate_index]
            if not np.any(valid_rows):
                continue
            candidate_next_indices = next_indices[valid_rows, candidate_index]
            candidate_continuation = continuation_by_next_volume[candidate_next_indices]
            candidate_values = (
                candidate_continuation
                + volume_deltas[valid_rows, candidate_index, np.newaxis] * prices[day][np.newaxis, :]
            )
            current_slice = current_values[valid_rows]
            better = candidate_values > current_slice
            current_values[valid_rows] = np.where(better, candidate_values, current_slice)
            if policy is not None:
                selected_slice = selected_next_indices[valid_rows]
                selected_next_indices[valid_rows] = np.where(
                    better,
                    candidate_next_indices[:, np.newaxis],
                    selected_slice,
                )
            if continuation_policy is not None:
                selected_continuation_slice = selected_continuation[valid_rows]
                selected_continuation[valid_rows] = np.where(
                    better,
                    candidate_continuation,
                    selected_continuation_slice,
                )

        if policy is not None:
            policy[day] = selected_next_indices
        if continuation_policy is not None:
            continuation_policy[day] = selected_continuation

        future_values = current_values

    start_index = _nearest_grid_index(grid, contract.start_volume)
    if valuation_price_paths is not None:
        return _evaluate_lsmc_policy_out_of_sample(
            valuation_price_paths,
            contract,
            basis,
            transitions,
            grid,
            start_index,
            coefficients_by_day,
            basis_scales,
            start_time=start_time,
            return_volume_paths=return_volume_paths,
            return_path_diagnostics=return_path_diagnostics,
        )

    volume_paths = _volume_paths_from_policy(grid, start_index, policy) if policy is not None else None
    daily_cashflows = None
    accumulated_cashflows = None
    continuation_values = None
    if return_path_diagnostics:
        if volume_paths is None or continuation_policy is None or policy is None:
            raise RuntimeError("path diagnostics require policy and volume paths")
        daily_cashflows = (volume_paths[:-1] - volume_paths[1:]) * prices
        accumulated_cashflows = np.cumsum(daily_cashflows, axis=0)
        continuation_values = _path_values_from_policy(start_index, policy, continuation_policy)
    return ValuationResult(
        value=float(np.mean(future_values[start_index])),
        runtime_seconds=time.perf_counter() - start_time,
        volume_paths=volume_paths,
        path_values=future_values[start_index].copy(),
        daily_cashflows=daily_cashflows,
        accumulated_cashflows=accumulated_cashflows,
        continuation_values=continuation_values,
    )


def _evaluate_lsmc_policy_out_of_sample(
    valuation_price_paths: NDArray[np.float64],
    contract: GasStorageContract,
    basis: BasisSpec,
    transitions: list[tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.bool_]]],
    grid: NDArray[np.float64],
    start_index: int,
    coefficients_by_day: list[NDArray[np.float64] | None],
    basis_scales: NDArray[np.float64],
    *,
    start_time: float,
    return_volume_paths: bool,
    return_path_diagnostics: bool,
) -> ValuationResult:
    """Apply trained continuation regressions to independent valuation paths."""

    prices = valuation_price_paths[:-1]
    n_days = contract.n_days
    n_paths = valuation_price_paths.shape[1]
    current_indices = np.full(n_paths, start_index, dtype=np.int32)
    path_values = np.zeros(n_paths, dtype=np.float64)
    volume_paths = np.empty((n_days + 1, n_paths), dtype=np.float64)
    volume_paths[0] = grid[current_indices]
    continuation_values = np.empty((n_days, n_paths), dtype=np.float64) if return_path_diagnostics else None

    for day in range(n_days):
        coefficients = coefficients_by_day[day]
        if coefficients is None:
            raise RuntimeError("missing trained continuation coefficients")
        design = build_basis_matrix(prices[day], basis, price_scale=float(basis_scales[day]))
        continuation_by_next_volume = (design @ coefficients).T
        next_indices, volume_deltas, valid = transitions[day]
        best_values = np.full(n_paths, -1.0e12, dtype=np.float64)
        best_next_indices = current_indices.copy()
        best_continuation = np.zeros(n_paths, dtype=np.float64)

        for candidate_index in range(next_indices.shape[1]):
            valid_paths = valid[current_indices, candidate_index]
            if not np.any(valid_paths):
                continue
            path_indexes = np.flatnonzero(valid_paths)
            candidate_next_indices = next_indices[current_indices[path_indexes], candidate_index]
            candidate_continuation = continuation_by_next_volume[candidate_next_indices, path_indexes]
            candidate_values = (
                candidate_continuation
                + volume_deltas[current_indices[path_indexes], candidate_index] * prices[day, path_indexes]
            )
            better = candidate_values > best_values[path_indexes]
            if not np.any(better):
                continue
            better_path_indexes = path_indexes[better]
            best_values[better_path_indexes] = candidate_values[better]
            best_next_indices[better_path_indexes] = candidate_next_indices[better]
            best_continuation[better_path_indexes] = candidate_continuation[better]

        cashflows = (grid[current_indices] - grid[best_next_indices]) * prices[day]
        path_values += cashflows
        current_indices = best_next_indices
        volume_paths[day + 1] = grid[current_indices]
        if continuation_values is not None:
            continuation_values[day] = best_continuation

    daily_cashflows = (volume_paths[:-1] - volume_paths[1:]) * prices if return_path_diagnostics else None
    accumulated_cashflows = np.cumsum(daily_cashflows, axis=0) if daily_cashflows is not None else None
    return ValuationResult(
        value=float(np.mean(path_values)),
        runtime_seconds=time.perf_counter() - start_time,
        volume_paths=volume_paths if return_volume_paths or return_path_diagnostics else None,
        path_values=path_values,
        daily_cashflows=daily_cashflows,
        accumulated_cashflows=accumulated_cashflows,
        continuation_values=continuation_values,
    )


def build_basis_matrix(
    prices: NDArray[np.float64],
    basis: BasisSpec,
    *,
    price_scale: float | None = None,
) -> NDArray[np.float64]:
    """Build a numerically scaled OLS design matrix."""

    scale = float(np.mean(prices)) if price_scale is None else float(price_scale)
    scaled = prices / scale - 1.0
    if basis.family == "power":
        return np.column_stack([scaled**degree for degree in range(basis.count)])
    if basis.family == "bspline":
        return _bspline_like_basis(scaled, basis.count)
    raise ValueError(f"unsupported basis family: {basis.family}")


def run_exhibit6_replication(
    *,
    runs: int = 10,
    path_counts: tuple[int, ...] = EXHIBIT6_PATH_COUNTS,
    basis_specs: tuple[BasisSpec, ...] = EXHIBIT6_BASIS_SPECS,
    sigma: float = 0.0945,
    base_seed: int = 20050610,
    out_of_sample: bool = False,
    valuation_seed_offset: int = 1_000_000,
    workers: int | None = None,
) -> list[Exhibit6Result]:
    """Run the Exhibit 6 basis/path-count grid and aggregate ten-run results."""

    contract = exhibit1_contract()
    transitions = _transition_candidates(contract, np.asarray(contract.volume_grid, dtype=np.float64))
    run_values: dict[tuple[str, int], list[float]] = {
        (basis.label, path_count): [] for basis in basis_specs for path_count in path_counts
    }
    run_std_values: dict[tuple[str, int], list[float]] = {
        (basis.label, path_count): [] for basis in basis_specs for path_count in path_counts
    }
    run_runtimes: dict[tuple[str, int], list[float]] = {
        (basis.label, path_count): [] for basis in basis_specs for path_count in path_counts
    }
    worker_count = workers if workers is not None else min(len(basis_specs), os.cpu_count() or 1)

    for path_count in path_counts:
        for run_index in range(runs):
            paths = simulate_exhibit_price_paths(
                contract=contract,
                n_paths=path_count,
                sigma=sigma,
                seed=base_seed + run_index,
            )
            valuation_paths = (
                simulate_exhibit_price_paths(
                    contract=contract,
                    n_paths=path_count,
                    sigma=sigma,
                    seed=base_seed + valuation_seed_offset + run_index,
                )
                if out_of_sample
                else None
            )
            if worker_count > 1 and len(basis_specs) > 1:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    basis_results = list(
                        executor.map(
                            lambda current_basis: (
                                current_basis,
                                value_storage_lsmc(
                                    paths,
                                    contract,
                                    current_basis,
                                    transitions,
                                    return_volume_paths=True,
                                    valuation_price_paths=valuation_paths,
                                ),
                            ),
                            basis_specs,
                        )
                    )
            else:
                basis_results = [
                    (
                        basis,
                        value_storage_lsmc(
                            paths,
                            contract,
                            basis,
                            transitions,
                            return_volume_paths=True,
                            valuation_price_paths=valuation_paths,
                        ),
                    )
                    for basis in basis_specs
                ]
            for basis, result in basis_results:
                key = (basis.label, path_count)
                run_values[key].append(result.value)
                if result.volume_paths is None:
                    raise RuntimeError("pathwise volumes were not returned")
                realized_values = _realized_storage_values(
                    result.volume_paths,
                    valuation_paths if valuation_paths is not None else paths,
                )
                run_std_values[key].append(
                    float(np.std(realized_values, ddof=1)) if len(realized_values) > 1 else 0.0
                )
                run_runtimes[key].append(result.runtime_seconds)

    results: list[Exhibit6Result] = []
    for basis in basis_specs:
        for path_count in path_counts:
            key = (basis.label, path_count)
            values = run_values[key]
            std_values = run_std_values[key]
            runtimes = run_runtimes[key]
            results.append(
                Exhibit6Result(
                    basis=basis.label,
                    path_count=path_count,
                    mean=float(np.mean(values)),
                    std_error=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    average_runtime_seconds=float(np.mean(runtimes)),
                    run_values=tuple(values),
                    run_std_values=tuple(std_values),
                )
            )
    return results


def run_exhibit6_out_of_sample_replication(
    *,
    runs: int = 10,
    path_counts: tuple[int, ...] = EXHIBIT6_PATH_COUNTS,
    basis_specs: tuple[BasisSpec, ...] = EXHIBIT6_BASIS_SPECS,
    sigma: float = 0.0945,
    base_seed: int = 20050610,
    valuation_seed_offset: int = 1_000_000,
    workers: int | None = None,
) -> list[Exhibit6Result]:
    """Run Exhibit 6 with regressions trained and valued on independent paths."""

    return run_exhibit6_replication(
        runs=runs,
        path_counts=path_counts,
        basis_specs=basis_specs,
        sigma=sigma,
        base_seed=base_seed,
        out_of_sample=True,
        valuation_seed_offset=valuation_seed_offset,
        workers=workers,
    )


def run_b6_convergence(
    *,
    runs: int = 10,
    path_counts: tuple[int, ...] = B6_CONVERGENCE_PATH_COUNTS,
    sigma: float = 0.0945,
    base_seed: int = 20050610,
) -> list[Exhibit6Result]:
    """Run the B6 basis on the requested MC path-count convergence grid."""

    return run_exhibit6_replication(
        runs=runs,
        path_counts=path_counts,
        basis_specs=(B6_CONVERGENCE_BASIS,),
        sigma=sigma,
        base_seed=base_seed,
        workers=1,
    )


def build_volume_plot_data(
    *,
    path_counts: tuple[int, ...] = VOLUME_PLOT_PATH_COUNTS,
    basis_specs: tuple[BasisSpec, ...] = EXHIBIT6_BASIS_SPECS,
    sigma: float = 0.0945,
    base_seed: int = 20050610,
) -> list[VolumePlotData]:
    """Generate one pathwise LSMC volume sample per basis/path-count pair."""

    contract = exhibit1_contract()
    transitions = _transition_candidates(contract, np.asarray(contract.volume_grid, dtype=np.float64))
    volume_plots: list[VolumePlotData] = []
    selected_basis_specs = tuple(basis for basis in basis_specs if basis.label in VOLUME_PLOT_BASIS_LABELS)
    for basis in selected_basis_specs:
        for path_count in path_counts:
            paths = simulate_exhibit_price_paths(
                contract=contract,
                n_paths=path_count,
                sigma=sigma,
                seed=base_seed,
            )
            result = value_storage_lsmc(paths, contract, basis, transitions, return_volume_paths=True)
            if result.volume_paths is None:
                raise RuntimeError("volume paths were not returned")
            volume_plots.append(VolumePlotData(basis=basis.label, path_count=path_count, volumes=result.volume_paths))
    return volume_plots


def build_price_plot_data(
    *,
    path_count: int = 50,
    sigma: float = 0.0945,
    base_seed: int = 20050610,
) -> PricePlotData:
    """Generate one pathwise gas-price sample for charting."""

    contract = exhibit1_contract()
    prices = simulate_exhibit_price_paths(
        contract=contract,
        n_paths=path_count,
        sigma=sigma,
        seed=base_seed,
    )
    return PricePlotData(path_count=path_count, prices=prices)


def build_storage_value_distribution_data(
    *,
    path_count: int = 500,
    basis: BasisSpec = B6_CONVERGENCE_BASIS,
    sigma: float = 0.0945,
    base_seed: int = 20050610,
) -> StorageValueDistributionData:
    """Generate one pathwise storage-value sample for a histogram."""

    contract = exhibit1_contract()
    transitions = _transition_candidates(contract, np.asarray(contract.volume_grid, dtype=np.float64))
    prices = simulate_exhibit_price_paths(
        contract=contract,
        n_paths=path_count,
        sigma=sigma,
        seed=base_seed,
    )
    result = value_storage_lsmc(prices, contract, basis, transitions, return_volume_paths=True)
    if result.volume_paths is None:
        raise RuntimeError("volume paths were not returned")
    realized_values = _realized_storage_values(result.volume_paths, prices)
    return StorageValueDistributionData(
        basis=basis,
        path_count=path_count,
        values=realized_values,
        intrinsic_value=intrinsic_contract_value(contract),
    )


def build_continuation_value_data(
    *,
    path_count: int = 50,
    basis: BasisSpec = BasisSpec("power", 4, "P3"),
    sigma: float = 0.0945,
    base_seed: int = 20050610,
) -> ContinuationValueData:
    """Generate pathwise accumulated cashflow and continuation values for P3."""

    contract = exhibit1_contract()
    transitions = _transition_candidates(contract, np.asarray(contract.volume_grid, dtype=np.float64))
    prices = simulate_exhibit_price_paths(
        contract=contract,
        n_paths=path_count,
        sigma=sigma,
        seed=base_seed,
    )
    result = value_storage_lsmc(
        prices,
        contract,
        basis,
        transitions,
        return_path_diagnostics=True,
    )
    if (
        result.volume_paths is None
        or result.daily_cashflows is None
        or result.accumulated_cashflows is None
        or result.continuation_values is None
    ):
        raise RuntimeError("continuation diagnostics were not returned")
    return ContinuationValueData(
        basis=basis,
        path_count=path_count,
        prices=prices,
        volumes=result.volume_paths,
        daily_cashflows=result.daily_cashflows,
        accumulated_cashflows=result.accumulated_cashflows,
        continuation_values=result.continuation_values,
    )


def write_exhibit6_workbook(
    path: Path,
    results: list[Exhibit6Result],
    volume_plots: list[VolumePlotData] | None = None,
    price_plot: PricePlotData | None = None,
    storage_distribution: StorageValueDistributionData | None = None,
    continuation_values: ContinuationValueData | None = None,
    convergence_b6_results: list[Exhibit6Result] | None = None,
    *,
    sigma: float = 0.0945,
    runs: int = 10,
) -> None:
    """Write reproduced and published Exhibit 6 values to an Excel workbook."""

    contract = exhibit1_contract()
    intrinsic_value = intrinsic_contract_value(contract)
    result_lookup = {(result.basis, result.path_count): result for result in results}
    headers = [
        "Basis",
        "MC run repetition",
        "Number of paths",
        "Mean MC runs",
        "Std MRC runs",
        "Intrinsic value",
        "Running time (sec)",
        "Basis functions",
    ]
    rows = []
    for basis in EXHIBIT6_BASIS_SPECS:
        for path_count in EXHIBIT6_PATH_COUNTS:
            result = result_lookup.get((basis.label, path_count))
            rows.append(
                [
                    basis.label,
                    runs,
                    path_count,
                    round(result.mean, 2) if result else "",
                    round(result.std_error, 2) if result else "",
                    round(intrinsic_value, 2),
                    round(result.average_runtime_seconds, 3) if result else "",
                    _basis_vector_label(basis),
                ]
            )
    notes = [
        ["Item", "Value"],
        ["Paper", "Boogert and de Jong, Gas Storage Valuation Using a Monte Carlo Method"],
        ["Contract", "Exhibit 1: 0-250000 MWh, start/end 100000 MWh, withdrawal 7500, injection 2500"],
        ["Method", "Backward dynamic program with OLS continuation values by day and next inventory grid point"],
        ["Price curve", "Synthetic seasonal TTF-style proxy; the paper forward curve is not included in this repository"],
        ["Paper results", "Excluded because the exact forward curve used in the paper is not available here"],
        ["Volatility", "High-volatility case sigma=9.45%, kappa=0.05"],
        ["Volume chart paths", "Generated from the first run seed for 50 paths only"],
    ]
    sheets: list[Sheet] = [("Exhibit 6", headers, rows), ("Notes", [], notes)]
    run_sheet_basis = B6_CONVERGENCE_BASIS
    run_sheet_results = [
        result
        for path_count in EXHIBIT6_PATH_COUNTS
        if (result := result_lookup.get(("B6", path_count))) is not None
    ]
    if not run_sheet_results:
        run_sheet_basis = BasisSpec("power", 4, "P3")
        run_sheet_results = [
            result
            for path_count in EXHIBIT6_PATH_COUNTS
            if (result := result_lookup.get(("P3", path_count))) is not None
        ]
    if run_sheet_results:
        sheets.append(_mc_run_values_sheet(run_sheet_results, run_sheet_basis, contract, sigma=sigma))
    if continuation_values:
        sheets.append(_continuation_value_sheet(continuation_values, contract, sigma=sigma))
    if convergence_b6_results:
        sheets.append(_convergence_b6_sheet(convergence_b6_results, contract, sigma=sigma, runs=runs))
    if storage_distribution:
        sheets.append(_storage_value_distribution_sheet(storage_distribution, contract, sigma=sigma))
    sheets.append(_forward_curve_sheet(contract, sigma=sigma))
    if price_plot:
        sheets.append(_price_plot_sheet(price_plot, contract, sigma=sigma))
    if volume_plots:
        sheets.extend(_volume_plot_sheets(volume_plots, contract, sigma=sigma))
    write_workbook(path, sheets)


def _mc_run_values_sheet(
    results: list[Exhibit6Result],
    basis: BasisSpec,
    contract: GasStorageContract,
    *,
    sigma: float,
) -> Sheet:
    sheet_name = "P3 all MC Runs" if basis.label == "P3" else f"{basis.label.lower()} all MC Runs"
    ordered_results = sorted(results, key=lambda item: item.path_count)
    headers = [
        label
        for result in ordered_results
        for label in (
            f"Mean MC estimates {result.path_count:,} paths",
            f"Std MC estimates {result.path_count:,} paths",
        )
    ]
    max_runs = max(len(result.run_values) for result in ordered_results)
    rows: list[list[object]] = _parameter_rows(
        contract,
        sigma=sigma,
        mc_paths=", ".join(f"{result.path_count:,}" for result in ordered_results),
        basis=basis,
    )
    rows.extend(
        [
        [
            "Mean MC runs",
        ]
        + [
            value
            for result in ordered_results
            for value in (
                round(float(np.mean(result.run_values)), 2) if result.run_values else round(result.mean, 2),
                "",
            )
        ],
        ["Std MRC runs"] + [value for result in ordered_results for value in (round(result.std_error, 2), "")],
        [],
        ["MC run"] + headers,
        ]
    )
    for run_index in range(max_runs):
        run_values: list[object] = []
        for result in ordered_results:
            if run_index < len(result.run_values):
                run_std_value = result.run_std_values[run_index] if run_index < len(result.run_std_values) else ""
                run_values.extend([result.run_values[run_index], run_std_value])
            else:
                run_values.extend(["", ""])
        rows.append(
            [run_index + 1] + run_values
        )
    run_data_start = 11
    run_data_end = run_data_start + max_runs - 1
    series = []
    for result_index, result in enumerate(ordered_results):
        estimate_column = _column_name(2 + result_index * 2)
        series.append(
            (
                f"Mean MC estimates {result.path_count:,} paths",
                f"'{sheet_name}'!$A${run_data_start}:$A${run_data_end}",
                f"'{sheet_name}'!${estimate_column}${run_data_start}:${estimate_column}${run_data_end}",
                "",
            )
        )
    vertical_lines = []
    for run_index in range(max_runs):
        run_estimates = [
            result.run_values[run_index]
            for result in ordered_results
            if run_index < len(result.run_values)
        ]
        if len(run_estimates) >= 2:
            vertical_lines.append((run_index + 1, min(run_estimates), max(run_estimates)))
    chart: ChartSpec = (
        "Mean MC Estimates by MC Run",
        2,
        9,
        series,
        "scatter",
        None,
        None,
        f"MC run (1\u2013{max_runs})",
        "Mean MC estimate",
        1,
        max_runs,
        "General",
        "#,##0_ ;[Red]\\-#,##0\\ ",
        True,
        {
            "scatter_style": "lineMarker",
            "excel_saved_chart_style": "p3_mc_runs",
            "vertical_lines": tuple(vertical_lines),
            "to_column": 21,
            "to_column_offset": 82550,
            "to_row": 29,
            "to_row_offset": 127000,
            "drawing_id": 4,
            "drawing_name": "Mean MC Estimates by Run",
            "drawing_ext_xml": (
                '<a:extLst><a:ext uri="{FF2B5EF4-FFF2-40B4-BE49-F238E27FC236}">'
                '<a16:creationId xmlns:a16="http://schemas.microsoft.com/office/drawing/2014/main" '
                'id="{C99A66DF-7960-5FBB-501D-81EC5EF3D275}"/></a:ext></a:extLst>'
            ),
        },
    )
    return (sheet_name, [], rows, [chart])


def _p3_run_values_sheet(results: list[Exhibit6Result], contract: GasStorageContract, *, sigma: float) -> Sheet:
    """Backward-compatible wrapper for the P3 all-runs sheet."""

    return _mc_run_values_sheet(results, BasisSpec("power", 4, "P3"), contract, sigma=sigma)


def _convergence_b6_sheet(
    results: list[Exhibit6Result],
    contract: GasStorageContract,
    *,
    sigma: float,
    runs: int,
) -> Sheet:
    sheet_name = "convergence b6"
    intrinsic_value = intrinsic_contract_value(contract)
    headers = [
        "Basis",
        "MC run repetition",
        "Number of paths",
        "Mean MC runs",
        "Std MRC runs",
        "Intrinsic value",
        "Running time (sec)",
        "Basis functions",
    ]
    rows = [
        [
            result.basis,
            runs,
            result.path_count,
            round(result.mean, 2),
            round(result.std_error, 2),
            round(intrinsic_value, 2),
            round(result.average_runtime_seconds, 3),
            _basis_vector_label(B6_CONVERGENCE_BASIS),
        ]
        for result in sorted(results, key=lambda item: item.path_count)
    ]
    data_start = 2
    data_end = data_start + len(rows) - 1
    chart: ChartSpec = (
        "B6 convergence by MC path count",
        1,
        10,
        [
            (
                "Mean MC runs",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                "18000",
                "1F4E79",
                "circle",
                None,
                "num",
            ),
        ],
        "line",
        None,
        None,
        "Number of paths",
        "Gas storage value",
        "#,##0",
        "#,##0",
        False,
        1,
        0,
        45,
    )
    return (sheet_name, headers, rows, [chart])


def _storage_value_distribution_sheet(
    distribution: StorageValueDistributionData,
    contract: GasStorageContract,
    *,
    sigma: float,
) -> Sheet:
    basis_label = distribution.basis.label.lower()
    sheet_name = f"Storage value distribution {basis_label}"
    counts, bin_edges = np.histogram(distribution.values, bins=20)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    intrinsic_index = int(np.argmin(np.abs(centers - distribution.intrinsic_value)))
    mean_value = float(np.mean(distribution.values))
    mean_index = int(np.argmin(np.abs(centers - mean_value)))
    marker_height = float(np.max(counts) if len(counts) else 1.0)

    intrinsic_markers = [0.0] * len(counts)
    mean_markers = [0.0] * len(counts)
    intrinsic_markers[intrinsic_index] = marker_height
    mean_markers[mean_index] = marker_height

    rows: list[list[object]] = _parameter_rows(
        contract,
        sigma=sigma,
        mc_paths=distribution.path_count,
        basis=distribution.basis,
    )
    rows.extend(
        [
            ["Mean storage value", round(mean_value, 2)],
            ["Intrinsic value", round(distribution.intrinsic_value, 2)],
            [],
            ["Storage value bin (EUR millions)", "Frequency", "Intrinsic value", "Mean storage value"],
        ]
    )
    for index, count in enumerate(counts):
        rows.append(
            [
                f"{bin_edges[index] / 1_000_000:.2f} - {bin_edges[index + 1] / 1_000_000:.2f}",
                int(count),
                intrinsic_markers[index],
                mean_markers[index],
            ]
        )

    header_row = len(rows) - len(counts)
    data_start = header_row + 1
    data_end = data_start + len(counts) - 1
    chart: ChartSpec = (
        f"{distribution.basis.label} storage value distribution across {distribution.path_count:,} paths",
        1,
        6,
        [
            (
                "Storage value frequency",
                f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                "",
                "9DC3E6",
            ),
            (
                "Intrinsic value",
                f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                "",
                "5B3A29",
            ),
            (
                "Mean storage value",
                f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                "",
                "1F4E79",
            ),
        ],
        "column",
        0,
        marker_height,
        "Storage value (EUR millions)",
        "Frequency",
        "0",
        "0",
        True,
    )
    return (sheet_name, [], rows, [chart])


def _continuation_value_sheet(
    continuation_data: ContinuationValueData,
    contract: GasStorageContract,
    *,
    sigma: float,
) -> Sheet:
    rows: list[list[object]] = _parameter_rows(
        contract,
        sigma=sigma,
        mc_paths=continuation_data.path_count,
        basis=continuation_data.basis,
    )
    charts: list[ChartSpec] = []
    dates = contract.dates
    n_days = contract.n_days
    saved_path_count = min(5, continuation_data.path_count)
    for path_index in range(saved_path_count):
        block_start = len(rows) + 1
        rows.append([f"Path {path_index + 1}"])
        rows.append(
            [
                "Day",
                "Date",
                "Date label",
                "Spot price S(t)",
                "Volume v_t",
                "Next volume v_{t+1}",
                "Daily cash flow",
                "Accumulated cash flow",
                "Continuation value",
            ]
        )
        for day_index in range(n_days):
            rows.append(
                [
                    day_index,
                    _excel_date_serial(dates[day_index]),
                    _quarterly_date_label(dates[day_index]),
                    float(continuation_data.prices[day_index, path_index]),
                    float(continuation_data.volumes[day_index, path_index]),
                    float(continuation_data.volumes[day_index + 1, path_index]),
                    float(continuation_data.daily_cashflows[day_index, path_index]),
                    float(continuation_data.accumulated_cashflows[day_index, path_index]),
                    float(continuation_data.continuation_values[day_index, path_index]),
                ]
            )
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + n_days - 1
        chart_row = 1 + path_index * 25
        charts.append(
            (
                f"Path {path_index + 1}: cash flow, continuation value, and volume",
                chart_row,
                11,
                [
                    (
                        "Accumulated cash flow",
                        f"'Continuation value P3'!$C${data_start}:$C${data_end}",
                        f"'Continuation value P3'!$H${data_start}:$H${data_end}",
                        "",
                        "26670",
                        "1F4E79",
                        "none",
                    ),
                    (
                        "Continuation value",
                        f"'Continuation value P3'!$C${data_start}:$C${data_end}",
                        f"'Continuation value P3'!$I${data_start}:$I${data_end}",
                        "",
                        "26670",
                        "5B3A29",
                        "none",
                    ),
                ],
                "line_secondary",
                None,
                None,
                "Date",
                "Cash flow / continuation value",
                "mmm\\-yyyy",
                "#,##0_ ;[Red]\\-#,##0\\ ",
                True,
                1,
                0,
                45,
                [
                    (
                        "Volume v_t",
                        f"'Continuation value P3'!$C${data_start}:$C${data_end}",
                        f"'Continuation value P3'!$E${data_start}:$E${data_end}",
                        "",
                        "26670",
                        "70AD47",
                        "none",
                    ),
                ],
                "Volume at t",
                "#,##0",
                {
                    "to_column": 24,
                    "to_column_offset": 361950,
                    "to_row": chart_row + 22,
                    "to_row_offset": 180974,
                    "primary_y_axis_format": "#,##0_ ;[Red]\\-#,##0\\ ",
                    "secondary_y_axis_format": "#,##0",
                    "legend_layout_xml": (
                        '<c:layout><c:manualLayout><c:xMode val="edge"/>'
                        '<c:yMode val="edge"/><c:x val="0.86"/>'
                        '<c:y val="0.16"/>'
                        '<c:w val="0.14"/>'
                        '<c:h val="0.18"/>'
                        "</c:manualLayout></c:layout>"
                    ),
                    "legend_overlay": False,
                },
            )
        )
    return ("Continuation value P3", [], rows, charts)


def _forward_curve_sheet(contract: GasStorageContract, *, sigma: float) -> Sheet:
    sheet_name = "Forward Curve"
    dates = contract.dates
    forward_curve = seasonal_ttf_proxy_curve(contract)
    calibrated_mu = calibrate_mu_to_forward_curve(
        forward_curve,
        kappa=GAS_KAPPA,
        sigma=sigma,
    )
    log_variance = _log_variance_path(len(forward_curve), kappa=GAS_KAPPA, sigma=sigma)
    rows: list[list[object]] = [["Day", "Date", "Date label", "Forward price", "calibrated mu(t)", "log variance v_day"]]
    for day_index, current_date in enumerate(dates):
        rows.append(
            [
                day_index,
                _excel_date_serial(current_date),
                _quarterly_date_label(current_date),
                float(forward_curve[day_index]),
                float(calibrated_mu[day_index]) if day_index < len(calibrated_mu) else "",
                float(log_variance[day_index]),
            ]
        )

    data_start = 2
    data_end = data_start + len(dates) - 1
    forward_chart: ChartSpec = (
        "Forward curve used for gas storage valuation",
        1,
        7,
        [
            (
                "Forward price",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                "",
                "18000",
                "1F4E79",
                "none",
            ),
        ],
        "line",
        None,
        None,
        "Date",
        "Price",
        "mmm-yyyy",
        "0.00",
        False,
        1,
        0,
        45,
    )
    mu_data_end = data_start + len(calibrated_mu) - 1
    mu_chart: ChartSpec = (
        "Calibrated mu(t)",
        19,
        7,
        [
            (
                "calibrated mu(t)",
                f"'{sheet_name}'!$C${data_start}:$C${mu_data_end}",
                f"'{sheet_name}'!$E${data_start}:$E${mu_data_end}",
                "",
                "18000",
                "5B3A29",
                "none",
            ),
        ],
        "line",
        None,
        None,
        "Date",
        "mu(t)",
        "mmm-yyyy",
        "0.0000",
        False,
        1,
        0,
        45,
    )
    variance_chart: ChartSpec = (
        "Log variance v_day",
        37,
        7,
        [
            (
                "log variance v_day",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                "",
                "18000",
                "8064A2",
                "none",
            ),
        ],
        "line",
        None,
        None,
        "Date",
        "v_day",
        "mmm-yyyy",
        "0.0000",
        False,
        1,
        0,
        45,
    )
    return (sheet_name, [], rows, [forward_chart, mu_chart, variance_chart])


def _price_plot_sheet(price_plot: PricePlotData, contract: GasStorageContract, *, sigma: float) -> Sheet:
    sheet_name = "Gas Prices 50 Paths"
    dates = contract.dates
    date_labels = [_quarterly_date_label(current_date) for current_date in dates]
    forward_curve = seasonal_ttf_proxy_curve(contract)
    rows: list[list[object]] = _parameter_rows(
        contract,
        sigma=sigma,
        mc_paths=price_plot.path_count,
        basis=BasisSpec("power", 4, "P3"),
    )
    rows.append(
        ["Day", "Date", "Date label", "Minimum S(t)", "Maximum S(t)", "Forward curve"]
        + [f"Path {path_index + 1}" for path_index in range(price_plot.prices.shape[1])]
    )
    header_row = len(rows)
    for day_index, current_date in enumerate(dates):
        path_prices = price_plot.prices[day_index]
        rows.append(
            [
                day_index,
                _excel_date_serial(current_date),
                date_labels[day_index],
                float(np.min(path_prices)),
                float(np.max(path_prices)),
                float(forward_curve[day_index]),
            ]
            + [float(price) for price in path_prices]
        )

    data_start = header_row + 1
    data_end = data_start + len(dates) - 1
    series = [
        (
            "",
            f"'{sheet_name}'!$C${data_start}:$C${data_end}",
            f"'{sheet_name}'!${_column_name(7 + path_index)}${data_start}:${_column_name(7 + path_index)}${data_end}",
            "",
            "4000",
            "D9D9D9",
            "none",
        )
        for path_index in range(price_plot.prices.shape[1])
    ]
    series.extend(
        [
            (
                "Minimum S(t)",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                "",
                "18000",
                "0070C0",
                "none",
            ),
            (
                "Maximum S(t)",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                "",
                "18000",
                "C00000",
                "none",
            ),
            (
                "Forward curve",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                "",
                "18000",
                "5B3A29",
                "none",
                "dash",
            ),
        ]
    )
    all_paths_chart: ChartSpec = (
        f"vol {sigma:.2%}: simulated gas prices across {price_plot.path_count} paths",
        1,
        8,
        series,
        "line",
        None,
        None,
        "Date",
        "S(t)",
        "mmm-yyyy",
        "0.00",
        False,
        1,
        price_plot.prices.shape[1],
        45,
    )
    selected_series = [
        (
            "Path 1",
            f"'{sheet_name}'!$C${data_start}:$C${data_end}",
            f"'{sheet_name}'!$G${data_start}:$G${data_end}",
            "",
            "12000",
            "70AD47",
            "none",
        ),
    ]
    if price_plot.prices.shape[1] >= 50:
        selected_series.append(
            (
                "Path 50",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                f"'{sheet_name}'!${_column_name(6 + 50)}${data_start}:${_column_name(6 + 50)}${data_end}",
                "",
                "12000",
                "7030A0",
                "none",
            )
        )
    selected_series.extend(
        [
            (
                "Minimum S(t)",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                "",
                "18000",
                "0070C0",
                "none",
            ),
            (
                "Maximum S(t)",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                "",
                "18000",
                "C00000",
                "none",
            ),
            (
                "Forward curve",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                "",
                "18000",
                "5B3A29",
                "none",
                "dash",
            ),
        ]
    )
    selected_paths_chart: ChartSpec = (
        f"vol {sigma:.2%}: selected gas price paths with min/max and forward curve",
        19,
        8,
        selected_series,
        "line",
        None,
        None,
        "Date",
        "S(t)",
        "mmm-yyyy",
        "0.00",
        True,
        1,
        0,
        45,
    )
    return (sheet_name, [], rows, [all_paths_chart, selected_paths_chart])


def _volume_plot_sheets(
    volume_plots: list[VolumePlotData],
    contract: GasStorageContract,
    *,
    sigma: float,
) -> list[Sheet]:
    sheets: list[Sheet] = []
    dates = contract.dates
    date_labels = [_quarterly_date_label(current_date) for current_date in dates]
    forward_curve = seasonal_ttf_proxy_curve(contract)
    intrinsic_volumes = intrinsic_schedule(forward_curve, contract)
    volatility_label = f"{sigma:.2%}"
    by_basis = {basis.label: [] for basis in EXHIBIT6_BASIS_SPECS}
    for plot in volume_plots:
        by_basis.setdefault(plot.basis, []).append(plot)

    for basis in EXHIBIT6_BASIS_SPECS:
        plots = sorted(by_basis.get(basis.label, []), key=lambda item: item.path_count)
        if not plots:
            continue
        sheet_name = f"Volume {basis.label}"
        rows: list[list[object]] = []
        charts: list[ChartSpec] = []
        chart_column = 8
        for plot_index, plot in enumerate(plots):
            block_start = len(rows) + 1
            rows.append([f"{plot.path_count:,} paths"])
            rows.append(
                ["Day", "Date", "Date label", "Minimum", "Maximum", "Intrinsic volume"]
                + [f"Path {path_index + 1}" for path_index in range(plot.volumes.shape[1])]
            )
            for day_index, current_date in enumerate(dates):
                path_volumes = plot.volumes[day_index]
                rows.append(
                    [
                        day_index,
                        _excel_date_serial(current_date),
                        date_labels[day_index],
                        float(np.min(path_volumes)),
                        float(np.max(path_volumes)),
                        float(intrinsic_volumes[day_index]),
                    ]
                    + [float(volume) for volume in path_volumes]
                )
            rows.append([])

            header_row = block_start + 1
            data_start = header_row + 1
            data_end = data_start + len(dates) - 1
            series = [
                (
                    "",
                    f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    f"'{sheet_name}'!${_column_name(7 + path_index)}${data_start}:${_column_name(7 + path_index)}${data_end}",
                    "",
                    "4000",
                    "D9D9D9",
                    "none",
                    None,
                )
                for path_index in range(plot.volumes.shape[1])
            ]
            series.extend(
                [
                    (
                        "Minimum",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                        f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                        "",
                        "18000",
                        "0070C0",
                        "none",
                        None,
                    ),
                    (
                        "Maximum",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                        "",
                        "18000",
                        "C00000",
                        "none",
                        None,
                    ),
                    (
                        "Intrinsic value",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                        f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                        "",
                        "18000",
                        "5B3A29",
                        "none",
                        "dash",
                    ),
                ]
            )
            charts.append(
                (
                    f"vol {volatility_label}, basis {basis.label} : volumes across {plot.path_count:,} paths",
                    1 + plot_index * 18,
                    chart_column,
                    series,
                    "line",
                    contract.min_volume,
                    contract.max_volume,
                    "Date",
                    "Volume (MWh)",
                    "mmm-yyyy",
                    "0",
                    False,
                    1,
                    plot.volumes.shape[1],
                    45,
                )
            )
        sheets.append((sheet_name, [], rows, charts))
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--path-counts", type=int, nargs="+", default=list(EXHIBIT6_PATH_COUNTS))
    parser.add_argument("--sigma", type=float, default=0.0945)
    parser.add_argument("--base-seed", type=int, default=20050610)
    parser.add_argument(
        "--out-of-sample",
        action="store_true",
        help="Train continuation regressions and value the policy on independent simulated paths.",
    )
    parser.add_argument(
        "--valuation-seed-offset",
        type=int,
        default=1_000_000,
        help="Seed offset for out-of-sample valuation paths.",
    )
    parser.add_argument(
        "--skip-convergence-b6",
        action="store_true",
        help="Do not run the B6 convergence grid for the convergence b6 sheet.",
    )
    args = parser.parse_args()

    results = run_exhibit6_replication(
        runs=args.runs,
        path_counts=tuple(args.path_counts),
        sigma=args.sigma,
        base_seed=args.base_seed,
        out_of_sample=args.out_of_sample,
        valuation_seed_offset=args.valuation_seed_offset,
    )
    volume_plots = build_volume_plot_data(
        path_counts=VOLUME_PLOT_PATH_COUNTS,
        sigma=args.sigma,
        base_seed=args.base_seed,
    )
    price_plot = build_price_plot_data(
        path_count=50,
        sigma=args.sigma,
        base_seed=args.base_seed,
    )
    storage_distribution = build_storage_value_distribution_data(
        path_count=500,
        sigma=args.sigma,
        base_seed=args.base_seed,
    )
    continuation_data = build_continuation_value_data(
        path_count=50,
        sigma=args.sigma,
        base_seed=args.base_seed,
    )
    convergence_b6_results = None
    if not args.skip_convergence_b6:
        convergence_b6_results = run_b6_convergence(
            runs=args.runs,
            sigma=args.sigma,
            base_seed=args.base_seed,
        )
    write_exhibit6_workbook(
        args.output,
        results,
        volume_plots,
        price_plot,
        storage_distribution,
        continuation_data,
        convergence_b6_results,
        sigma=args.sigma,
        runs=args.runs,
    )
    print(f"Excel workbook saved to {args.output.resolve()}")


def _bspline_like_basis(scaled_prices: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    """Compact cubic spline-style basis without external dependencies."""

    if count < 4:
        raise ValueError("bspline basis count must be at least 4")
    columns = [np.ones_like(scaled_prices), scaled_prices, scaled_prices**2, scaled_prices**3]
    knot_count = count - 4
    if knot_count:
        knots = np.quantile(scaled_prices, np.linspace(0.15, 0.85, knot_count))
        columns.extend(np.maximum(scaled_prices - knot, 0.0) ** 3 for knot in knots)
    return np.column_stack(columns[:count])


def _validate_paths(price_paths: NDArray[np.float64], contract: GasStorageContract) -> None:
    if price_paths.ndim != 2:
        raise ValueError("price_paths must have shape (n_days + 1, n_paths)")
    if price_paths.shape[0] != contract.n_days + 1:
        raise ValueError("price_paths first dimension must equal n_days + 1")
    if price_paths.shape[1] < 1:
        raise ValueError("price_paths must contain at least one path")
    if np.any(price_paths <= 0.0):
        raise ValueError("price_paths must be strictly positive")


def _nearest_grid_index(grid: NDArray[np.float64], volume: float) -> int:
    return int(np.argmin(np.abs(grid - volume)))


def _volume_paths_from_policy(
    grid: NDArray[np.float64],
    start_index: int,
    policy: NDArray[np.int16],
) -> NDArray[np.float64]:
    n_days, _, n_paths = policy.shape
    path_indexes = np.arange(n_paths)
    current_indices = np.full(n_paths, start_index, dtype=np.int32)
    volumes = np.empty((n_days + 1, n_paths), dtype=np.float64)
    volumes[0] = grid[current_indices]
    for day in range(n_days):
        current_indices = policy[day, current_indices, path_indexes].astype(np.int32)
        volumes[day + 1] = grid[current_indices]
    return volumes


def _path_values_from_policy(
    start_index: int,
    policy: NDArray[np.int16],
    values_by_state: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_days, _, n_paths = policy.shape
    path_indexes = np.arange(n_paths)
    current_indices = np.full(n_paths, start_index, dtype=np.int32)
    path_values = np.empty((n_days, n_paths), dtype=np.float64)
    for day in range(n_days):
        path_values[day] = values_by_state[day, current_indices, path_indexes]
        current_indices = policy[day, current_indices, path_indexes].astype(np.int32)
    return path_values


def _realized_storage_values(
    volume_paths: NDArray[np.float64],
    price_paths: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return pathwise realized storage cashflows under a volume policy."""

    return np.sum((volume_paths[:-1] - volume_paths[1:]) * price_paths[:-1], axis=0)


def _transition_candidates(
    contract: GasStorageContract,
    grid: NDArray[np.float64],
) -> list[tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.bool_]]]:
    transitions: list[tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.bool_]]] = []
    max_candidates = int(contract.max_withdrawal_per_day / contract.volume_step) + int(
        contract.max_injection_per_day / contract.volume_step
    ) + 1
    for day in range(contract.n_days):
        days_remaining_after_step = contract.n_days - day - 1
        next_indices = np.zeros((len(grid), max_candidates), dtype=np.int32)
        volume_deltas = np.zeros((len(grid), max_candidates), dtype=np.float64)
        valid = np.zeros((len(grid), max_candidates), dtype=np.bool_)
        for volume_index, volume in enumerate(grid):
            feasible = contract.feasible_next_volumes(float(volume), days_remaining_after_step)
            for candidate_index, next_volume in enumerate(feasible):
                next_index = _nearest_grid_index(grid, next_volume)
                next_indices[volume_index, candidate_index] = next_index
                volume_deltas[volume_index, candidate_index] = volume - grid[next_index]
                valid[volume_index, candidate_index] = True
        transitions.append((next_indices, volume_deltas, valid))
    return transitions


def _column_name(column_index: int) -> str:
    name = ""
    while column_index:
        column_index, remainder = divmod(column_index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _parameter_rows(
    contract: GasStorageContract,
    *,
    sigma: float,
    mc_paths: int | str,
    basis: BasisSpec,
) -> list[list[object]]:
    return [
        ["Polynomials", _basis_vector_label(basis)],
        ["kappa", GAS_KAPPA],
        ["sigma", sigma],
        ["Number of MC paths", mc_paths],
        ["Number of time steps", contract.n_days],
        [],
    ]


def _basis_vector_label(basis: BasisSpec) -> str:
    if basis.family == "power":
        return "[" + ", ".join("1" if degree == 0 else f"x^{degree}" for degree in range(basis.count)) + "]"
    if basis.family == "bspline":
        columns = ["1", "x", "x^2", "x^3"]
        columns.extend(f"(x-k{index})_+^3" for index in range(1, max(0, basis.count - 4) + 1))
        return "[" + ", ".join(columns[: basis.count]) + "], k_j = day-specific scaled-price quantiles"
    return f"{basis.family} basis with {basis.count} functions"


def _log_variance_path(length: int, *, kappa: float, sigma: float) -> NDArray[np.float64]:
    variances = np.empty(length, dtype=np.float64)
    variances[0] = 0.0
    for index in range(1, length):
        variances[index] = (1.0 - kappa) ** 2 * variances[index - 1] + sigma * sigma
    return variances


def _vector_label(values: NDArray[np.float64]) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def _excel_date_serial(value: date) -> int:
    return (value - date(1899, 12, 30)).days


def _quarterly_date_label(value: date) -> str:
    if value.day == 1 and value.month in (1, 4, 7, 10):
        return value.strftime("%b-%Y")
    return ""


if __name__ == "__main__":
    main()
