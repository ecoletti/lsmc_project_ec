"""GBM LSMC variance-reduction diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lsmc_testing.black_scholes import american_put_price_barone_adesi_whaley, european_put_price
from lsmc_testing.excel import ChartSpec, Sheet, write_workbook
from lsmc_testing.pricing import laguerre_basis, put_payoff
from lsmc_testing.simulation import simulate_gbm_paths


STRIKE = 40.0
RATE = 0.0
MATURITY = 1.0
NUMBER_OF_TIME_STEPS = 250
N_PATHS = 100_000
PATH_COUNT_RANGE = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000)
LAGUERRE_DEGREE = 2
PATH_COUNT_REGRESSION_PATHS = 50_000
BASE_SEED = 12345
GBM_VOLATILITY_RANGE = (0.05, 0.10, 0.20, 0.30)
SPOT_LEVELS = (36.0, 38.0, 40.0, 42.0, 44.0)
DEFAULT_OUTPUT_PATH = Path("output") / "lsmc_gbm_variance_reduction.xlsx"


@dataclass(frozen=True)
class EstimatorResult:
    price: float
    standard_error: float
    variance: float


def lsm_present_values(
    paths: NDArray[np.float64],
    strike: float,
    rate: float,
    maturity: float,
    laguerre_degree: int = LAGUERRE_DEGREE,
    regression_max_paths: int | None = None,
) -> NDArray[np.float64]:
    if paths.ndim != 2:
        raise ValueError("paths must have shape (n_steps + 1, n_paths)")

    n_steps = paths.shape[0] - 1
    dt = maturity / n_steps
    discount = exp(-rate * dt)

    cashflows = put_payoff(paths[-1], strike)
    for step in range(n_steps - 1, 0, -1):
        cashflows *= discount
        exercise_value = put_payoff(paths[step], strike)
        in_the_money = exercise_value > 0.0
        in_the_money_count = np.count_nonzero(in_the_money)
        if in_the_money_count < 4:
            continue

        in_the_money_indices = np.flatnonzero(in_the_money)
        if regression_max_paths is not None and in_the_money_count > regression_max_paths:
            regression_indices = in_the_money_indices[:regression_max_paths]
        else:
            regression_indices = in_the_money_indices

        scaled_spot = paths[step, regression_indices] / strike
        regression_x = laguerre_basis(scaled_spot, degree=laguerre_degree)
        regression_y = cashflows[regression_indices]
        coefficients, *_ = np.linalg.lstsq(regression_x, regression_y, rcond=None)
        continuation_value = laguerre_basis(paths[step, in_the_money_indices] / strike, degree=laguerre_degree) @ coefficients

        exercise_now = exercise_value[in_the_money_indices] > continuation_value
        exercise_indices = in_the_money_indices[exercise_now]
        cashflows[exercise_indices] = exercise_value[exercise_indices]

    return cashflows * discount


def standard_estimator(values: NDArray[np.float64]) -> EstimatorResult:
    variance = float(np.var(values, ddof=1))
    return EstimatorResult(
        price=round(float(np.mean(values)), 8),
        standard_error=round(float(np.std(values, ddof=1) / sqrt(values.size)), 8),
        variance=variance,
    )


def control_variate_estimator(
    american_values: NDArray[np.float64],
    european_values: NDArray[np.float64],
    european_exact: float,
) -> EstimatorResult:
    control_variance = np.var(european_values, ddof=1)
    if control_variance <= 0.0:
        return standard_estimator(american_values)

    beta = np.cov(american_values, european_values, ddof=1)[0, 1] / control_variance
    adjusted_values = american_values + beta * (european_exact - european_values)
    return standard_estimator(adjusted_values)


def european_present_values(paths: NDArray[np.float64], strike: float, rate: float, maturity: float) -> NDArray[np.float64]:
    return exp(-rate * maturity) * put_payoff(paths[-1], strike)


def simulate_paths(s0: float, sigma: float, antithetic: bool, seed: int, rate: float = RATE, n_paths: int = N_PATHS) -> NDArray[np.float64]:
    return simulate_gbm_paths(
        s0=s0,
        rate=rate,
        sigma=sigma,
        maturity=MATURITY,
        steps_per_year=NUMBER_OF_TIME_STEPS,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )


def calculate_case(
    s0: float,
    sigma: float,
    seed: int,
    rate: float = RATE,
    laguerre_degree: int = LAGUERRE_DEGREE,
) -> tuple[float, EstimatorResult, EstimatorResult, EstimatorResult]:
    plain_paths = simulate_paths(s0, sigma, antithetic=False, seed=seed, rate=rate)
    plain_american_values = lsm_present_values(plain_paths, STRIKE, rate, MATURITY, laguerre_degree)
    european_values = european_present_values(plain_paths, STRIKE, rate, MATURITY)
    european_exact = european_put_price(s0, STRIKE, rate, sigma, MATURITY)
    correlation = round(float(np.corrcoef(plain_american_values, european_values)[0, 1]), 8)

    antithetic_paths = simulate_paths(s0, sigma, antithetic=True, seed=seed + 1_000_000, rate=rate)
    antithetic_american_values = lsm_present_values(antithetic_paths, STRIKE, rate, MATURITY, laguerre_degree)

    return (
        correlation,
        standard_estimator(plain_american_values),
        standard_estimator(antithetic_american_values),
        control_variate_estimator(plain_american_values, european_values, european_exact),
    )


def calculate_plain_and_control_case(
    s0: float,
    sigma: float,
    seed: int,
    rate: float,
    n_paths: int = N_PATHS,
    laguerre_degree: int = LAGUERRE_DEGREE,
) -> tuple[EstimatorResult, EstimatorResult]:
    plain_paths = simulate_paths(s0, sigma, antithetic=False, seed=seed, rate=rate, n_paths=n_paths)
    plain_american_values = lsm_present_values(plain_paths, STRIKE, rate, MATURITY, laguerre_degree)
    european_values = european_present_values(plain_paths, STRIKE, rate, MATURITY)
    european_exact = european_put_price(s0, STRIKE, rate, sigma, MATURITY)
    return (
        standard_estimator(plain_american_values),
        control_variate_estimator(plain_american_values, european_values, european_exact),
    )


def calculate_baw_comparison_blocks(rate: float) -> dict[float, list[list[Any]]]:
    blocks: dict[float, list[list[Any]]] = {sigma: [] for sigma in GBM_VOLATILITY_RANGE}
    for spot_index, s0 in enumerate(SPOT_LEVELS):
        for sigma_index, sigma in enumerate(GBM_VOLATILITY_RANGE):
            seed = BASE_SEED + spot_index * 100 + sigma_index
            plain, control = calculate_plain_and_control_case(s0, sigma, seed, rate=rate)
            european_price = european_put_price(s0, STRIKE, rate, sigma, MATURITY)
            baw_price = american_put_price_barone_adesi_whaley(s0, STRIKE, rate, sigma, MATURITY)
            blocks[sigma].append(
                [
                    s0,
                    plain.price,
                    plain.standard_error,
                    control.price,
                    control.standard_error,
                    round(european_price, 8),
                    round(baw_price, 8),
                ]
            )
    return blocks


def calculate_path_count_comparison_blocks(rate: float, laguerre_degree: int = LAGUERRE_DEGREE) -> dict[float, list[list[Any]]]:
    s0 = 40.0
    blocks: dict[float, list[list[Any]]] = {sigma: [] for sigma in GBM_VOLATILITY_RANGE}
    for sigma_index, sigma in enumerate(GBM_VOLATILITY_RANGE):
        european_price = european_put_price(s0, STRIKE, rate, sigma, MATURITY)
        baw_price = american_put_price_barone_adesi_whaley(s0, STRIKE, rate, sigma, MATURITY)
        max_paths = max(PATH_COUNT_RANGE)
        paths = simulate_paths(
            s0,
            sigma,
            antithetic=False,
            seed=BASE_SEED + 10_000 + sigma_index * 1_000,
            rate=rate,
            n_paths=max_paths,
        )
        american_values = lsm_present_values(
            paths,
            STRIKE,
            rate,
            MATURITY,
            laguerre_degree,
            regression_max_paths=PATH_COUNT_REGRESSION_PATHS,
        )
        european_values = european_present_values(paths, STRIKE, rate, MATURITY)
        for path_index, n_paths in enumerate(PATH_COUNT_RANGE):
            plain_american_values = american_values[:n_paths]
            european_value_subset = european_values[:n_paths]
            plain = standard_estimator(plain_american_values)
            control = control_variate_estimator(plain_american_values, european_value_subset, european_price)
            blocks[sigma].append(
                [
                    n_paths,
                    plain.price,
                    plain.standard_error,
                    control.price,
                    control.standard_error,
                    round(european_price, 8),
                    round(baw_price, 8),
                ]
            )
    return blocks


def calculate_results(rate: float = RATE) -> tuple[list[list[Any]], dict[float, list[list[Any]]], dict[float, list[list[Any]]], dict[float, list[list[Any]]]]:
    correlations = []
    estimator_blocks: dict[float, list[list[Any]]] = {sigma: [] for sigma in GBM_VOLATILITY_RANGE}
    diagnostic_blocks: dict[float, list[list[Any]]] = {sigma: [] for sigma in GBM_VOLATILITY_RANGE}
    baw_comparison_blocks: dict[float, list[list[Any]]] = {sigma: [] for sigma in GBM_VOLATILITY_RANGE}

    for spot_index, s0 in enumerate(SPOT_LEVELS):
        correlation_row = [s0]
        for sigma_index, sigma in enumerate(GBM_VOLATILITY_RANGE):
            seed = BASE_SEED + spot_index * 100 + sigma_index
            correlation, plain, antithetic, control = calculate_case(s0, sigma, seed, rate=rate)
            european_price = european_put_price(s0, STRIKE, rate, sigma, MATURITY)
            baw_price = american_put_price_barone_adesi_whaley(s0, STRIKE, rate, sigma, MATURITY)
            correlation_row.append(correlation)
            variance_ratio = control.variance / plain.variance if plain.variance > 0.0 else ""
            antithetic_variance_ratio = antithetic.variance / plain.variance if plain.variance > 0.0 else ""
            diagnostic_blocks[sigma].append(
                [
                    s0,
                    antithetic_variance_ratio,
                    variance_ratio,
                    1.0 - correlation * correlation,
                    plain.variance,
                    antithetic.variance,
                    control.variance,
                ]
            )
            estimator_blocks[sigma].append(
                [
                    s0,
                    plain.price,
                    plain.standard_error,
                    antithetic.price,
                    antithetic.standard_error,
                    control.price,
                    control.standard_error,
                ]
            )
            baw_comparison_blocks[sigma].append([s0, plain.price, plain.standard_error, control.price, control.standard_error, round(european_price, 8), round(baw_price, 8)])
        correlations.append(correlation_row)

    return correlations, estimator_blocks, diagnostic_blocks, baw_comparison_blocks


def padded_bounds(values: list[float], padding_fraction: float = 0.25, floor: float | None = None, ceiling: float | None = None) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    padding = max((max_value - min_value) * padding_fraction, abs(max_value) * 0.05, 0.01)
    lower = min_value - padding
    upper = max_value + padding
    if floor is not None:
        lower = max(floor, lower)
    if ceiling is not None:
        upper = min(ceiling, upper)
    if lower == upper:
        upper = lower + 1.0
    return round(lower, 8), round(upper, 8)


def tight_bounds(values: list[float], padding_fraction: float = 0.05, floor: float | None = None) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    padding = max((max_value - min_value) * padding_fraction, 0.01)
    lower = min_value - padding
    upper = max_value + padding
    if floor is not None:
        lower = max(floor, lower)
    if lower == upper:
        upper = lower + 1.0
    return round(lower, 8), round(upper, 8)


def workbook_input_rows(rate: float = RATE, laguerre_degree: int = LAGUERRE_DEGREE) -> list[list[Any]]:
    return [
        ["Strike", STRIKE],
        ["Maturity", MATURITY],
        ["Rate", rate],
        ["Dividend", 0.0],
        ["Number of MC paths", N_PATHS],
        ["Number of time steps", NUMBER_OF_TIME_STEPS],
        ["Polynomials", laguerre_polynomial_description(laguerre_degree)],
        [],
    ]


def build_correlation_sheet(rows: list[list[Any]], sheet_name: str = "American European Corr 1", rate: float = RATE) -> Sheet:
    headers = ["Spot level"] + [f"Vol {sigma:.2f}" for sigma in GBM_VOLATILITY_RANGE]
    input_rows = workbook_input_rows(rate)
    output_rows = input_rows + [headers] + rows
    header_row = len(input_rows) + 1
    data_start_row = header_row + 1
    data_end_row = data_start_row + len(rows) - 1
    correlation_values = [value for row in rows for value in row[1:]]
    y_axis_min, y_axis_max = padded_bounds(correlation_values, floor=0.0, ceiling=1.0)
    charts: list[ChartSpec] = [
        (
            "Correlation between pathwise American and European put PVs",
            1,
            8,
            [
                (
                    f"Spot {row[0]:.0f}",
                    f"'{sheet_name}'!$B${header_row}:$E${header_row}",
                    f"'{sheet_name}'!$B${row_index}:$E${row_index}",
                    "",
                )
                for row_index, row in enumerate(rows, start=data_start_row)
            ],
            "line",
            y_axis_min,
            y_axis_max,
            "Volatility",
            "Correlation",
        ),
        (
            "Correlation vs spot by volatility",
            18,
            8,
            [
                (
                    f"Vol {sigma:.2f}",
                    f"'{sheet_name}'!$A${data_start_row}:$A${data_end_row}",
                    f"'{sheet_name}'!${chr(66 + sigma_index)}${data_start_row}:${chr(66 + sigma_index)}${data_end_row}",
                    "",
                )
                for sigma_index, sigma in enumerate(GBM_VOLATILITY_RANGE)
            ],
            "line",
            y_axis_min,
            y_axis_max,
            "Spot",
            "Correlation",
        ),
    ]
    return (sheet_name, [], output_rows, charts)


def build_estimator_sheet(
    blocks: dict[float, list[list[Any]]],
    diagnostic_blocks: dict[float, list[list[Any]]],
    sheet_name: str = "Variances comparison 1",
    rate: float = RATE,
) -> Sheet:
    headers = [
        "Spot level",
        "No VR Price",
        "No VR s.e.",
        "Antithetic Price",
        "Antithetic s.e.",
        "Control Variate Price",
        "Control Variate s.e.",
    ]
    rows: list[list[Any]] = workbook_input_rows(rate)
    charts: list[ChartSpec] = []
    estimator_ranges: dict[float, tuple[int, int]] = {}

    for block_index, sigma in enumerate(GBM_VOLATILITY_RANGE):
        block_start = len(rows) + 1
        rows.append([f"Volatility {sigma:.2f}"])
        rows.append(headers)
        for row in blocks[sigma]:
            control_price = row[3]
            control_se = row[4]
            baw_price = row[6]
            difference = control_price - baw_price
            rows.append(
                row
                + [
                    difference / control_price if control_price != 0.0 else "",
                    difference / control_se if control_se > 0.0 else "",
                ]
            )

        data_start = block_start + 2
        data_end = data_start + len(SPOT_LEVELS) - 1
        estimator_ranges[sigma] = (data_start, data_end)
        rows.append([])

    diagnostic_headers = [
        "Spot level",
        "Antithetic variance / No VR variance",
        "Control Variate variance / No VR variance",
        "1 - correlation^2",
        "No VR variance",
        "Antithetic variance",
        "Control Variate variance",
    ]
    rows.append([])
    rows.append(["Variance Ratio Diagnostics"])
    diagnostic_ranges: dict[float, tuple[int, int]] = {}
    for block_index, sigma in enumerate(GBM_VOLATILITY_RANGE):
        block_start = len(rows) + 1
        rows.append([f"Volatility {sigma:.2f}"])
        rows.append(diagnostic_headers)
        rows.extend(diagnostic_blocks[sigma])

        data_start = block_start + 2
        data_end = data_start + len(SPOT_LEVELS) - 1
        diagnostic_ranges[sigma] = (data_start, data_end)
        ratio_x_values = [float(row[3]) for row in diagnostic_blocks[sigma]]
        ratio_y_values = [float(row[2]) for row in diagnostic_blocks[sigma]]
        x_axis_min, x_axis_max = padded_bounds(ratio_x_values, floor=0.0)
        y_axis_min, y_axis_max = padded_bounds(ratio_y_values, floor=0.0)
        charts.append(
            (
                f"Control variate variance ratio vs 1-rho^2, volatility {sigma:.2f}",
                1 + block_index * 16,
                10,
                [
                    (
                        f"Vol {sigma:.2f}",
                        f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                        "",
                    )
                ],
                "scatter",
                y_axis_min,
                y_axis_max,
                "1-corr^2",
                "Variance ratio",
                x_axis_min,
                x_axis_max,
            )
        )
        rows.append([])

    for block_index, sigma in enumerate(GBM_VOLATILITY_RANGE):
        data_start, data_end = estimator_ranges[sigma]
        standard_error_values = [
            float(value)
            for row in blocks[sigma]
            for value in (row[2], row[4], row[6])
        ]
        y_axis_min, y_axis_max = padded_bounds(standard_error_values, floor=0.0)
        charts.append(
            (
                f"MC standard error - {sigma:.2f}",
                1 + block_index * 16,
                20,
                [
                    (
                        "No variance reduction",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                        "",
                    ),
                    (
                        "Antithetic",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                        "",
                    ),
                    (
                        "Control variate",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$G${data_start}:$G${data_end}",
                        "",
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Spot",
                "Standard error",
            )
        )

    return (sheet_name, [], rows, charts)


def build_baw_comparison_sheet(blocks: dict[float, list[list[Any]]], sheet_name: str = "Control Variate MC vs BAW 1", rate: float = RATE) -> Sheet:
    headers = [
        "Spot",
        "No VR MC Price",
        "No VR MC s.e.",
        "Control Variate MC Price",
        "Control Variate MC s.e.",
        "European Put Price",
        "BAW Approx Price",
    ]
    rows: list[list[Any]] = workbook_input_rows(rate)
    charts: list[ChartSpec] = []

    for block_index, sigma in enumerate(GBM_VOLATILITY_RANGE):
        block_start = len(rows) + 1
        rows.append([f"Volatility {sigma:.2f}"])
        rows.append(headers)
        for row in blocks[sigma]:
            control_price = row[3]
            control_se = row[4]
            baw_price = row[6]
            difference = control_price - baw_price
            rows.append(
                row
                + [
                    difference / control_price if control_price != 0.0 else "",
                    difference / control_se if control_se > 0.0 else "",
                ]
            )

        data_start = block_start + 2
        data_end = data_start + len(SPOT_LEVELS) - 1
        price_values = [
            float(value)
            for row in blocks[sigma]
            for value in (row[1] - row[2], row[1] + row[2], row[3] - row[4], row[3] + row[4], row[5], row[6])
        ]
        y_axis_min, y_axis_max = padded_bounds(price_values, floor=0.0)
        charts.append(
            (
                f"MC and analytic put prices, volatility {sigma:.2f}",
                1 + block_index * 16,
                10,
                [
                    (
                        "No VR MC",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    ),
                    (
                        "Control Variate MC",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                    ),
                    (
                        "European put",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                        "",
                    ),
                    (
                        "BAW approximation",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$G${data_start}:$G${data_end}",
                        "",
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Spot",
                "Price",
            )
        )
        rows.append([])

    diagnostic_headers = [
        "Spot",
        "Control Variate MC Price",
        "Control Variate MC s.e.",
        "European Put Price",
        "(European - CV MC) / CV MC s.e.",
        "European within 2 s.e.",
        "BAW Approx Price",
        "(BAW - CV MC) / CV MC s.e.",
        "BAW within 2 s.e.",
    ]
    rows.append([])
    rows.append(["Within 2 Control Variate MC Standard Errors"])
    for sigma in GBM_VOLATILITY_RANGE:
        rows.append([f"Volatility {sigma:.2f}"])
        rows.append(diagnostic_headers)
        for row in blocks[sigma]:
            spot, _, _, control_price, control_se, european_price, baw_price = row
            european_difference = european_price - control_price
            baw_difference = baw_price - control_price
            european_standard_error_ratio = european_difference / control_se if control_se > 0.0 else ""
            baw_standard_error_ratio = baw_difference / control_se if control_se > 0.0 else ""
            threshold = 2.0 * control_se
            rows.append(
                [
                    spot,
                    control_price,
                    control_se,
                    european_price,
                    european_standard_error_ratio,
                    "Yes" if abs(european_difference) <= threshold else "No",
                    baw_price,
                    baw_standard_error_ratio,
                    "Yes" if abs(baw_difference) <= threshold else "No",
                ]
            )
        rows.append([])

    return (sheet_name, [], rows, charts)


def laguerre_polynomial_description(degree: int) -> str:
    polynomials = [
        "e^(-x/2)",
        "e^(-x/2)*(1-x)",
        "e^(-x/2)*(1-2x+x^(2)/2)",
        "e^(-x/2)*(1-3x+3x^(2)/2-x^(3)/6)",
        "e^(-x/2)*(1-4x+3x^(2)-2x^(3)/3+x^(4)/24)",
    ]
    return "[" + ", ".join(polynomials[: degree + 1]) + "]"


def build_path_count_comparison_sheet(
    blocks: dict[float, list[list[Any]]],
    sheet_name: str = "MC Paths vs BAW 1 dg 2",
    rate: float = 0.0,
    laguerre_degree: int = 2,
) -> Sheet:
    headers = [
        "Number of MC paths",
        "No VR MC Price",
        "No VR MC s.e.",
        "Control Variate MC Price",
        "Control Variate MC s.e.",
        "European Put Price",
        "BAW Approx Price",
        "(CV MC Price - BAW Approx Price) / CV MC Price",
        "(CV MC Price - BAW Approx Price) / CV MC s.e.",
    ]
    rows: list[list[Any]] = [
        ["Strike", STRIKE],
        ["Maturity", MATURITY],
        ["Rate", rate],
        ["Dividend", 0.0],
        ["S0", 40.0],
        ["Number of MC paths", ", ".join(f"{path_count:,}" for path_count in PATH_COUNT_RANGE)],
        ["Polynomials", laguerre_polynomial_description(laguerre_degree)],
        ["Number of time steps", NUMBER_OF_TIME_STEPS],
    ]
    charts: list[ChartSpec] = []

    for block_index, sigma in enumerate(GBM_VOLATILITY_RANGE):
        block_start = len(rows) + 1
        rows.append([f"Volatility {sigma:.2f}"])
        rows.append(headers)
        for row in blocks[sigma]:
            control_price = row[3]
            control_se = row[4]
            baw_price = row[6]
            difference = control_price - baw_price
            rows.append(
                row
                + [
                    difference / control_price if control_price != 0.0 else "",
                    difference / control_se if control_se > 0.0 else "",
                ]
            )

        data_start = block_start + 2
        data_end = data_start + len(PATH_COUNT_RANGE) - 1
        price_values = [
            float(value)
            for row in blocks[sigma]
            for value in (row[1] - row[2], row[1] + row[2], row[3] - row[4], row[3] + row[4], row[6])
        ]
        y_axis_min, y_axis_max = tight_bounds(price_values, floor=0.0)
        charts.append(
            (
                f"MC prices by path count, volatility {sigma:.2f}",
                1 + block_index * 18,
                10,
                [
                    (
                        "No VR MC",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    ),
                    (
                        "Control Variate MC",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                    ),
                    (
                        "BAW approximation",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$G${data_start}:$G${data_end}",
                        "",
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Number of MC paths",
                "Price",
                "0",
                "0.00",
            )
        )
        rows.append([])

    return (sheet_name, [], rows, charts)


def build_convergence_sheets_by_volatility(
    rate_0_blocks_by_degree: dict[int, dict[float, list[list[Any]]]],
    rate_006_blocks_by_degree: dict[int, dict[float, list[list[Any]]]],
) -> list[Sheet]:
    sheets: list[Sheet] = []
    polynomial_description = "; ".join(
        f"dg{degree}: {laguerre_polynomial_description(degree)}"
        for degree in (2, 3, 4)
    )

    for sigma in GBM_VOLATILITY_RANGE:
        sheet_name = f"MC CV minus BAW conv vol={sigma:.2f}"
        rows: list[list[Any]] = [
            ["Strike", STRIKE],
            ["Maturity", MATURITY],
            ["Rate", "0.00% and 0.06%"],
            ["Dividend", 0.0],
            ["S0", 40.0],
            ["Number of MC paths", ", ".join(f"{path_count:,}" for path_count in PATH_COUNT_RANGE)],
            ["Polynomials", polynomial_description],
            [],
        ]
        charts: list[ChartSpec] = []
        table_ranges: dict[str, tuple[int, int]] = {}

        for table_title, blocks_by_degree in [
            ("Rate 0.00%", rate_0_blocks_by_degree),
            ("Rate 0.06%", rate_006_blocks_by_degree),
        ]:
            table_start = len(rows) + 1
            headers = ["Number of MC paths"]
            for degree in (2, 3, 4):
                headers.extend(
                    [
                        f"dg{degree} Vol {sigma:.2f} (CV MC Price - BAW Approx Price) / CV MC s.e.",
                        f"dg{degree} Vol {sigma:.2f} Control Variate MC s.e.",
                    ]
                )

            rows.append([table_title])
            rows.append(headers)
            for row_index, path_count in enumerate(PATH_COUNT_RANGE):
                output_row: list[Any] = [path_count]
                for degree in (2, 3, 4):
                    source_row = blocks_by_degree[degree][sigma][row_index]
                    control_price = source_row[3]
                    control_se = source_row[4]
                    baw_price = source_row[6]
                    difference = control_price - baw_price
                    output_row.extend(
                        [
                            difference / control_se if control_se > 0.0 else "",
                            control_se,
                        ]
                    )
                rows.append(output_row)
            table_ranges[table_title] = (table_start + 2, table_start + 1 + len(PATH_COUNT_RANGE))
            rows.append([])

        chart_data_start = len(rows) + 1
        rows.append(["512,000 paths chart data"])
        rows.append(
            [
                "Degree",
                "Rate 0.00% (CV MC Price - BAW Approx Price) / CV MC s.e.",
                "Rate 0.00% Control Variate MC s.e.",
                "Rate 0.06% (CV MC Price - BAW Approx Price) / CV MC s.e.",
                "Rate 0.06% Control Variate MC s.e.",
            ]
        )
        for degree in (2, 3, 4):
            row_rate_0 = rate_0_blocks_by_degree[degree][sigma][-1]
            row_rate_006 = rate_006_blocks_by_degree[degree][sigma][-1]
            control_price_0 = row_rate_0[3]
            control_se_0 = row_rate_0[4]
            baw_price_0 = row_rate_0[6]
            control_price_006 = row_rate_006[3]
            control_se_006 = row_rate_006[4]
            baw_price_006 = row_rate_006[6]
            rows.append(
                [
                    degree,
                    (control_price_0 - baw_price_0) / control_se_0 if control_se_0 > 0.0 else "",
                    control_se_0,
                    (control_price_006 - baw_price_006) / control_se_006 if control_se_006 > 0.0 else "",
                    control_se_006,
                ]
            )

        chart_values_start = chart_data_start + 2
        chart_values_end = chart_values_start + 2
        ratio_values = [
            float(value)
            for row in rows[chart_values_start - 1 : chart_values_end]
            for value in (row[1], row[3])
        ]
        se_values = [
            float(value)
            for row in rows[chart_values_start - 1 : chart_values_end]
            for value in (row[2], row[4])
        ]
        ratio_min, ratio_max = padded_bounds(ratio_values)
        se_min, se_max = padded_bounds(se_values, floor=0.0)
        charts.append(
            (
                f"Vol {sigma:.2f} CV minus BAW over s.e. at 512,000 paths",
                1,
                10,
                [
                    (
                        "Rate 0.00%",
                        f"'{sheet_name}'!$A${chart_values_start}:$A${chart_values_end}",
                        f"'{sheet_name}'!$B${chart_values_start}:$B${chart_values_end}",
                        "",
                    ),
                    (
                        "Rate 0.06%",
                        f"'{sheet_name}'!$A${chart_values_start}:$A${chart_values_end}",
                        f"'{sheet_name}'!$D${chart_values_start}:$D${chart_values_end}",
                        "",
                    ),
                ],
                "line",
                ratio_min,
                ratio_max,
                "Laguerre degree",
                "(CV MC Price - BAW Approx Price) / CV MC s.e.",
                "0",
                "0.00",
            )
        )
        charts.append(
            (
                f"Vol {sigma:.2f} Control Variate MC s.e. at 512,000 paths",
                1,
                20,
                [
                    (
                        "Rate 0.00%",
                        f"'{sheet_name}'!$A${chart_values_start}:$A${chart_values_end}",
                        f"'{sheet_name}'!$C${chart_values_start}:$C${chart_values_end}",
                        "",
                    ),
                    (
                        "Rate 0.06%",
                        f"'{sheet_name}'!$A${chart_values_start}:$A${chart_values_end}",
                        f"'{sheet_name}'!$E${chart_values_start}:$E${chart_values_end}",
                        "",
                    ),
                ],
                "line",
                se_min,
                se_max,
                "Laguerre degree",
                "Control Variate MC s.e.",
                "0",
                "0.00",
            )
        )

        sheets.append((sheet_name, [], rows, charts))

    return sheets


def build_workbook_sheets() -> list[Sheet]:
    print(f"Calculating GBM variance-reduction results with {N_PATHS:,} paths...")
    correlations, estimator_blocks, diagnostic_blocks, baw_comparison_blocks = calculate_results()
    correlations_rate_006, estimator_blocks_rate_006, diagnostic_blocks_rate_006, _ = calculate_results(rate=0.06)
    baw_comparison_blocks_rate_006 = calculate_baw_comparison_blocks(rate=0.06)
    path_count_comparison_blocks = calculate_path_count_comparison_blocks(rate=0.0)
    path_count_comparison_blocks_rate_006 = calculate_path_count_comparison_blocks(rate=0.06)
    path_count_comparison_blocks_degree_3 = calculate_path_count_comparison_blocks(rate=0.0, laguerre_degree=3)
    path_count_comparison_blocks_rate_006_degree_3 = calculate_path_count_comparison_blocks(rate=0.06, laguerre_degree=3)
    path_count_comparison_blocks_degree_4 = calculate_path_count_comparison_blocks(rate=0.0, laguerre_degree=4)
    path_count_comparison_blocks_rate_006_degree_4 = calculate_path_count_comparison_blocks(rate=0.06, laguerre_degree=4)
    sheets: list[Sheet] = [
        build_correlation_sheet(correlations),
        build_estimator_sheet(estimator_blocks, diagnostic_blocks),
        build_correlation_sheet(correlations_rate_006, sheet_name="American European Corr 2", rate=0.06),
        build_estimator_sheet(
            estimator_blocks_rate_006,
            diagnostic_blocks_rate_006,
            sheet_name="Variances comparison 2",
            rate=0.06,
        ),
        build_baw_comparison_sheet(baw_comparison_blocks),
        build_baw_comparison_sheet(baw_comparison_blocks_rate_006, sheet_name="Control Variate MC vs BAW 2", rate=0.06),
        build_path_count_comparison_sheet(path_count_comparison_blocks),
        build_path_count_comparison_sheet(
            path_count_comparison_blocks_rate_006,
            sheet_name="MC Paths vs BAW 2 dg 2",
            rate=0.06,
        ),
        build_path_count_comparison_sheet(
            path_count_comparison_blocks_degree_3,
            sheet_name="MC Paths vs BAW 1 dg 3",
            rate=0.0,
            laguerre_degree=3,
        ),
        build_path_count_comparison_sheet(
            path_count_comparison_blocks_rate_006_degree_3,
            sheet_name="MC Paths vs BAW 2 dg 3",
            rate=0.06,
            laguerre_degree=3,
        ),
        build_path_count_comparison_sheet(
            path_count_comparison_blocks_degree_4,
            sheet_name="MC Paths vs BAW 1 dg 4",
            rate=0.0,
            laguerre_degree=4,
        ),
        build_path_count_comparison_sheet(
            path_count_comparison_blocks_rate_006_degree_4,
            sheet_name="MC Paths vs BAW 2 dg 4",
            rate=0.06,
            laguerre_degree=4,
        ),
    ]
    sheets.extend(
        build_convergence_sheets_by_volatility(
            {
                2: path_count_comparison_blocks,
                3: path_count_comparison_blocks_degree_3,
                4: path_count_comparison_blocks_degree_4,
            },
            {
                2: path_count_comparison_blocks_rate_006,
                3: path_count_comparison_blocks_rate_006_degree_3,
                4: path_count_comparison_blocks_rate_006_degree_4,
            },
        )
    )
    return sheets


def main() -> None:
    write_workbook(DEFAULT_OUTPUT_PATH, build_workbook_sheets())
    print(f"Excel file saved to {DEFAULT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
