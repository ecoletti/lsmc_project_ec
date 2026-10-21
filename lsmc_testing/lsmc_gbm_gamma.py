"""GBM LSMC Gamma sensitivity diagnostics."""

from __future__ import annotations

from math import exp, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from lsmc_testing.excel import ChartSpec, Sheet, write_workbook
from lsmc_testing.pricing import laguerre_basis, put_payoff
from lsmc_testing.simulation import simulate_gbm_paths


STRIKE = 40.0
MATURITY = 1.0
RATES = (0.0, 0.06)
DIVIDEND = 0.0
N_PATHS = 100_000
NUMBER_OF_TIME_STEPS = 250
LAGUERRE_DEGREES = (2, 3, 4)
SPOT_LEVELS = (36.0, 38.0, 40.0, 42.0, 44.0)
VOLATILITY_LEVELS = (0.05, 0.10, 0.20, 0.30)
BUMP_FRACTIONS = (0.01, 0.005)
BASE_SEED = 20260928
VALUATION_SEED_OFFSET = 1_000_000
DEFAULT_OUTPUT_PATH = Path("output") / "Doc to share" / "Gamma GBM out-of-sample.xlsx"
GammaResults = dict[tuple[float, float, int], dict[float, list[list[Any]]]]


def laguerre_polynomial_description(degree: int) -> str:
    polynomials = [
        "e^(-x/2)",
        "e^(-x/2)*(1-x)",
        "e^(-x/2)*(1-2x+x^(2)/2)",
        "e^(-x/2)*(1-3x+3x^(2)/2-x^(3)/6)",
        "e^(-x/2)*(1-4x+3x^(2)-2x^(3)/3+x^(4)/24)",
    ]
    return "[" + ", ".join(polynomials[: degree + 1]) + "]"


def lsm_out_of_sample_present_values(
    training_paths: np.ndarray,
    valuation_paths: np.ndarray,
    strike: float,
    rate: float,
    maturity: float,
    laguerre_degree: int,
) -> np.ndarray:
    if training_paths.ndim != 2 or valuation_paths.ndim != 2:
        raise ValueError("paths must have shape (n_steps + 1, n_paths)")
    if training_paths.shape[0] != valuation_paths.shape[0]:
        raise ValueError("training_paths and valuation_paths must use the same time grid")

    n_steps = training_paths.shape[0] - 1
    dt = maturity / n_steps
    discount = exp(-rate * dt)
    continuation_coefficients: list[np.ndarray | None] = [None] * (n_steps + 1)

    training_cashflows = put_payoff(training_paths[-1], strike)
    for step in range(n_steps - 1, 0, -1):
        training_cashflows *= discount
        exercise_value = put_payoff(training_paths[step], strike)
        in_the_money = exercise_value > 0.0
        if np.count_nonzero(in_the_money) < 4:
            continue

        regression_x = laguerre_basis(training_paths[step, in_the_money] / strike, degree=laguerre_degree)
        regression_y = training_cashflows[in_the_money]
        coefficients, *_ = np.linalg.lstsq(regression_x, regression_y, rcond=None)
        continuation_coefficients[step] = coefficients
        continuation_value = regression_x @ coefficients

        exercise_now = exercise_value[in_the_money] > continuation_value
        exercise_indices = np.flatnonzero(in_the_money)[exercise_now]
        training_cashflows[exercise_indices] = exercise_value[exercise_indices]

    valuation_cashflows = put_payoff(valuation_paths[-1], strike)
    for step in range(n_steps - 1, 0, -1):
        valuation_cashflows *= discount
        coefficients = continuation_coefficients[step]
        if coefficients is None:
            continue

        exercise_value = put_payoff(valuation_paths[step], strike)
        in_the_money = exercise_value > 0.0
        if not np.any(in_the_money):
            continue

        continuation_value = (
            laguerre_basis(valuation_paths[step, in_the_money] / strike, degree=laguerre_degree)
            @ coefficients
        )
        exercise_now = exercise_value[in_the_money] > continuation_value
        exercise_indices = np.flatnonzero(in_the_money)[exercise_now]
        valuation_cashflows[exercise_indices] = exercise_value[exercise_indices]

    return valuation_cashflows * discount


def calculate_gamma_case(
    *,
    spot: float,
    rate: float,
    sigma: float,
    seed: int,
    bump_fraction: float,
    laguerre_degree: int,
) -> tuple[float, float]:
    bump = spot * bump_fraction
    if bump <= 0.0 or spot - bump <= 0.0:
        raise ValueError("spot and bump_fraction must produce positive bumped spots")

    training_base_paths = simulate_gbm_paths(
        s0=spot,
        rate=rate,
        sigma=sigma,
        maturity=MATURITY,
        steps_per_year=NUMBER_OF_TIME_STEPS,
        n_paths=N_PATHS,
        seed=seed,
        antithetic=True,
    )
    training_growth_factors = training_base_paths / spot

    valuation_base_paths = simulate_gbm_paths(
        s0=spot,
        rate=rate,
        sigma=sigma,
        maturity=MATURITY,
        steps_per_year=NUMBER_OF_TIME_STEPS,
        n_paths=N_PATHS,
        seed=seed + VALUATION_SEED_OFFSET,
        antithetic=True,
    )
    valuation_growth_factors = valuation_base_paths / spot

    values_up = lsm_out_of_sample_present_values(
        training_growth_factors * (spot + bump),
        valuation_growth_factors * (spot + bump),
        STRIKE,
        rate,
        MATURITY,
        laguerre_degree=laguerre_degree,
    )
    values_mid = lsm_out_of_sample_present_values(
        training_base_paths,
        valuation_base_paths,
        STRIKE,
        rate,
        MATURITY,
        laguerre_degree=laguerre_degree,
    )
    values_down = lsm_out_of_sample_present_values(
        training_growth_factors * (spot - bump),
        valuation_growth_factors * (spot - bump),
        STRIKE,
        rate,
        MATURITY,
        laguerre_degree=laguerre_degree,
    )

    gamma_values = (values_up - 2.0 * values_mid + values_down) / (bump * bump)
    gamma = float(np.mean(gamma_values))
    standard_error = float(np.std(gamma_values, ddof=1) / sqrt(N_PATHS))
    return round(gamma, 10), round(standard_error, 10)


def calculate_gamma_results(rate: float, bump_fraction: float, laguerre_degree: int) -> dict[float, list[list[Any]]]:
    results: dict[float, list[list[Any]]] = {}

    for volatility_index, sigma in enumerate(VOLATILITY_LEVELS):
        rows: list[list[Any]] = []
        for spot_index, spot in enumerate(SPOT_LEVELS):
            print(
                "Calculating Gamma: "
                f"rate={rate:g}, bump={bump_fraction:g}, degree={laguerre_degree}, "
                f"volatility={sigma:.2f}, spot={spot:.0f}"
            )
            seed = (
                BASE_SEED
                + laguerre_degree * 10_000_000
                + int(rate * 1_000_000)
                + volatility_index * 1_000
                + spot_index
            )
            gamma, standard_error = calculate_gamma_case(
                spot=spot,
                rate=rate,
                sigma=sigma,
                seed=seed,
                bump_fraction=bump_fraction,
                laguerre_degree=laguerre_degree,
            )
            rows.append([spot, gamma, standard_error])
        results[sigma] = rows

    return results


def input_rows(rate: float, bump_fraction: float, laguerre_degree: int) -> list[list[Any]]:
    return [
        ["Strike", STRIKE],
        ["Maturity", MATURITY],
        ["Rate", rate],
        ["Dividend", DIVIDEND],
        ["Number of training MC paths", N_PATHS],
        ["Number of valuation MC paths", N_PATHS],
        ["Number of time steps", NUMBER_OF_TIME_STEPS],
        ["Polynomials", laguerre_polynomial_description(laguerre_degree)],
        ["Gamma bump", bump_fraction],
        ["Polynomial degree", laguerre_degree],
        ["Valuation method", "Out-of-sample"],
        [],
    ]


def padded_bounds(values: list[float], padding_fraction: float = 0.15) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    padding = max((max_value - min_value) * padding_fraction, abs(max_value) * 0.05, 0.000001)
    lower = min_value - padding
    upper = max_value + padding
    if lower == upper:
        upper = lower + 1.0
    return round(lower, 10), round(upper, 10)


def column_name(column_index: int) -> str:
    name = ""
    while column_index:
        column_index, remainder = divmod(column_index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def gamma_by_volatility_rows(gamma_results: dict[float, list[list[Any]]]) -> list[list[Any]]:
    headers = ["Volatility"]
    for spot in SPOT_LEVELS:
        headers.extend([f"Spot {spot:.0f} Gamma", f"Spot {spot:.0f} s.e."])
    rows: list[list[Any]] = [headers]

    for sigma in VOLATILITY_LEVELS:
        values_by_spot = {row[0]: row for row in gamma_results[sigma]}
        row_values: list[Any] = [sigma]
        for spot in SPOT_LEVELS:
            _, gamma, standard_error = values_by_spot[spot]
            row_values.extend([gamma, standard_error])
        rows.append(row_values)

    return rows


def build_gamma_sheet(
    rate: float,
    bump_fraction: float,
    laguerre_degree: int,
    gamma_results: dict[float, list[list[Any]]],
) -> Sheet:
    sheet_name = f"Gamma GBM r={rate:g} bp={bump_fraction:g} dg{laguerre_degree}"
    rows = input_rows(rate, bump_fraction, laguerre_degree)
    charts: list[ChartSpec] = []

    for volatility_index, sigma in enumerate(VOLATILITY_LEVELS):
        block_start = len(rows) + 1
        rows.append([f"Volatility {sigma:.2f}"])
        rows.append(["Spot", "Gamma", "Gamma standard error"])
        rows.extend(gamma_results[sigma])
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + len(SPOT_LEVELS) - 1
        gamma_values = [float(row[1]) for row in gamma_results[sigma]]
        standard_errors = [float(row[2]) for row in gamma_results[sigma]]
        bounds_values = [
            value
            for gamma, standard_error in zip(gamma_values, standard_errors)
            for value in (gamma - standard_error, gamma + standard_error)
        ]
        y_axis_min, y_axis_max = padded_bounds(bounds_values)
        charts.append(
            (
                f"Gamma profile with +/- 1 s.e., volatility {sigma:.2f}",
                1 + volatility_index * 16,
                6,
                [
                    (
                        f"Vol {sigma:.2f}",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Spot",
                "Gamma",
                "0",
                "0.000000",
            )
        )

    summary_start = len(rows) + 1
    summary_rows = gamma_by_volatility_rows(gamma_results)
    rows.append(["Gamma by volatility, strike 40"])
    rows.extend(summary_rows)
    summary_header_row = summary_start + 1
    summary_data_start = summary_header_row + 1
    summary_data_end = summary_data_start + len(VOLATILITY_LEVELS) - 1
    summary_values = [
        float(value)
        for row in summary_rows[1:]
        for value in row[1::2]
    ]
    summary_bounds_values = [
        value
        for row in summary_rows[1:]
        for gamma, standard_error in zip(row[1::2], row[2::2])
        for value in (float(gamma) - float(standard_error), float(gamma) + float(standard_error))
    ]
    summary_y_min, summary_y_max = padded_bounds(summary_bounds_values or summary_values)
    charts.append(
        (
            "Gamma against volatility, strike 40",
            1,
            18,
            [
                (
                    f"Spot {spot:.0f}",
                    f"'{sheet_name}'!$A${summary_data_start}:$A${summary_data_end}",
                    (
                        f"'{sheet_name}'!${column_name(2 + spot_index * 2)}"
                        f"${summary_data_start}:${column_name(2 + spot_index * 2)}${summary_data_end}"
                    ),
                    (
                        f"'{sheet_name}'!${column_name(3 + spot_index * 2)}"
                        f"${summary_data_start}:${column_name(3 + spot_index * 2)}${summary_data_end}"
                    ),
                )
                for spot_index, spot in enumerate(SPOT_LEVELS)
            ],
            "line",
            summary_y_min,
            summary_y_max,
            "Volatility",
            "Gamma",
            "0.00",
            "0.000000",
        )
    )

    return (sheet_name, [], rows, charts)


def build_error_sheet(rate: float, laguerre_degree: int, all_results: GammaResults) -> Sheet:
    sheet_name = f"Gamma error r={rate:g} dg{laguerre_degree}"
    rows: list[list[Any]] = [
        ["Strike", STRIKE],
        ["Maturity", MATURITY],
        ["Rate", rate],
        ["Dividend", DIVIDEND],
        ["Number of training MC paths", N_PATHS],
        ["Number of valuation MC paths", N_PATHS],
        ["Number of time steps", NUMBER_OF_TIME_STEPS],
        ["Polynomials", laguerre_polynomial_description(laguerre_degree)],
        ["Gamma bumps", ", ".join(f"{bump:g}" for bump in BUMP_FRACTIONS)],
        ["Polynomial degree", laguerre_degree],
        ["Valuation method", "Out-of-sample"],
        [],
    ]
    charts: list[ChartSpec] = []

    for volatility_index, sigma in enumerate(VOLATILITY_LEVELS):
        block_start = len(rows) + 1
        rows.append([f"Volatility {sigma:.2f}"])
        rows.append(["Bump"] + [f"Spot {spot:.0f}" for spot in SPOT_LEVELS])
        for bump_fraction in BUMP_FRACTIONS:
            se_by_spot = {
                row[0]: row[2]
                for row in all_results[(rate, bump_fraction, laguerre_degree)][sigma]
            }
            rows.append([bump_fraction] + [se_by_spot[spot] for spot in SPOT_LEVELS])
        rows.append([])

        header_row = block_start + 1
        data_start = header_row + 1
        data_end = data_start + len(BUMP_FRACTIONS) - 1
        error_values = [
            float(value)
            for row in rows[data_start - 1 : data_end]
            for value in row[1:]
        ]
        y_axis_min, y_axis_max = padded_bounds(error_values)
        charts.append(
            (
                f"Gamma standard error vs bump, volatility {sigma:.2f}",
                1 + volatility_index * 16,
                9,
                [
                    (
                        f"Spot {spot:.0f}",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!${chr(66 + spot_index)}${data_start}:${chr(66 + spot_index)}${data_end}",
                        "",
                    )
                    for spot_index, spot in enumerate(SPOT_LEVELS)
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Bump size",
                "Gamma standard error",
                "0.000",
                "0.000000",
            )
        )

    return (sheet_name, [], rows, charts)


def calculate_all_gamma_results() -> GammaResults:
    return {
        (rate, bump_fraction, laguerre_degree): calculate_gamma_results(
            rate,
            bump_fraction,
            laguerre_degree,
        )
        for laguerre_degree in LAGUERRE_DEGREES
        for bump_fraction in BUMP_FRACTIONS
        for rate in RATES
    }


def build_workbook_sheets() -> list[Sheet]:
    all_results = calculate_all_gamma_results()
    sheets = [
        build_gamma_sheet(
            rate,
            bump_fraction,
            laguerre_degree,
            all_results[(rate, bump_fraction, laguerre_degree)],
        )
        for laguerre_degree in LAGUERRE_DEGREES
        for rate in RATES
        for bump_fraction in BUMP_FRACTIONS
    ]
    sheets.extend(
        build_error_sheet(rate, laguerre_degree, all_results)
        for laguerre_degree in LAGUERRE_DEGREES
        for rate in RATES
    )
    return sheets


def main() -> None:
    write_workbook(DEFAULT_OUTPUT_PATH, build_workbook_sheets())
    print(f"Excel file saved to {DEFAULT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
