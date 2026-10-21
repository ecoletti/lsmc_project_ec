"""Least-squares Monte Carlo pricing for an American put."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LSMResult:
    price: float
    standard_error: float
    exercise_fraction: float


def put_payoff(spot: NDArray[np.float64], strike: float) -> NDArray[np.float64]:
    return np.maximum(strike - spot, 0.0)


def laguerre_basis(x: NDArray[np.float64], degree: int = 2) -> NDArray[np.float64]:
    """Constant plus weighted Laguerre polynomials up to ``degree``."""
    if degree < 0:
        raise ValueError("degree must be non-negative")

    weight = np.exp(-0.5 * x)
    polynomials = [np.ones_like(x), 1.0 - x]
    for order in range(2, degree + 1):
        next_polynomial = ((2 * order - 1 - x) * polynomials[-1] - (order - 1) * polynomials[-2]) / order
        polynomials.append(next_polynomial)

    return np.column_stack([np.ones_like(x)] + [weight * polynomial for polynomial in polynomials[: degree + 1]])


def price_american_put_lsm(
    paths: NDArray[np.float64],
    *,
    strike: float,
    rate: float,
    maturity: float,
    laguerre_degree: int = 2,
    valuation_paths: NDArray[np.float64] | None = None,
    control_variate_values: NDArray[np.float64] | None = None,
    control_variate_mean: float | None = None,
) -> LSMResult:
    """Price an American put using backward Longstaff-Schwartz regression."""
    if paths.ndim != 2:
        raise ValueError("paths must have shape (n_steps + 1, n_paths)")

    n_steps = paths.shape[0] - 1
    n_paths = paths.shape[1]
    dt = maturity / n_steps
    discount = exp(-rate * dt)
    continuation_coefficients: list[NDArray[np.float64] | None] = [None] * (n_steps + 1)

    cashflows = put_payoff(paths[-1], strike)
    exercise_step = np.full(n_paths, n_steps, dtype=np.int32)

    for step in range(n_steps - 1, 0, -1):
        cashflows *= discount
        exercise_value = put_payoff(paths[step], strike)
        in_the_money = exercise_value > 0.0
        if np.count_nonzero(in_the_money) < 4:
            continue

        scaled_spot = paths[step, in_the_money] / strike
        regression_x = laguerre_basis(scaled_spot, degree=laguerre_degree)
        regression_y = cashflows[in_the_money]
        coefficients, *_ = np.linalg.lstsq(regression_x, regression_y, rcond=None)
        continuation_coefficients[step] = coefficients
        continuation_value = regression_x @ coefficients

        exercise_now_itm = exercise_value[in_the_money] > continuation_value
        exercise_indices = np.flatnonzero(in_the_money)[exercise_now_itm]
        cashflows[exercise_indices] = exercise_value[exercise_indices]
        exercise_step[exercise_indices] = step

    if valuation_paths is not None:
        if valuation_paths.ndim != 2:
            raise ValueError("valuation_paths must have shape (n_steps + 1, n_paths)")
        if valuation_paths.shape[0] != paths.shape[0]:
            raise ValueError("valuation_paths must use the same time grid as paths")
        n_paths = valuation_paths.shape[1]
        cashflows = put_payoff(valuation_paths[-1], strike)
        exercise_step = np.full(n_paths, n_steps, dtype=np.int32)
        for step in range(n_steps - 1, 0, -1):
            cashflows *= discount
            coefficients = continuation_coefficients[step]
            if coefficients is None:
                continue
            exercise_value = put_payoff(valuation_paths[step], strike)
            in_the_money = exercise_value > 0.0
            if not np.any(in_the_money):
                continue

            scaled_spot = valuation_paths[step, in_the_money] / strike
            continuation_value = laguerre_basis(scaled_spot, degree=laguerre_degree) @ coefficients
            exercise_now_itm = exercise_value[in_the_money] > continuation_value
            exercise_indices = np.flatnonzero(in_the_money)[exercise_now_itm]
            cashflows[exercise_indices] = exercise_value[exercise_indices]
            exercise_step[exercise_indices] = step

    present_values = cashflows * discount
    if control_variate_values is not None and control_variate_mean is not None:
        if control_variate_values.shape != present_values.shape:
            raise ValueError("control_variate_values must have one value per path")
        control_variance = np.var(control_variate_values, ddof=1)
        if control_variance > 0.0:
            covariance = np.cov(present_values, control_variate_values, ddof=1)[0, 1]
            beta = covariance / control_variance
            present_values = present_values + beta * (control_variate_mean - control_variate_values)
    return LSMResult(
        price=float(np.mean(present_values)),
        standard_error=float(np.std(present_values, ddof=1) / sqrt(n_paths)),
        exercise_fraction=float(np.mean(exercise_step < n_steps)),
    )
