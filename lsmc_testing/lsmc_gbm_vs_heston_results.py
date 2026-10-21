"""Generate GBM versus spot-scaled Heston LSMC result workbooks."""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from lsmc_testing.black_scholes import european_put_price
from lsmc_testing.excel import ChartSpec, Sheet, write_workbook
from lsmc_testing.pricing import put_payoff
from lsmc_testing.pricing import price_american_put_lsm as raw_price_american_put_lsm
from lsmc_testing.simulation import (
    simulate_gbm_paths,
    simulate_heston_spot_scaled_vol_of_vol_paths,
)


S0 = 1.0
RATE = 0.0
MATURITY = 1.0
VARIANCE_REDUCTION_METHOD = "antithetic"
VARIANCE_REDUCTION_METHODS = ("none", "antithetic", "control_variate")
STEPS_PER_YEAR = 50
STRIKES = (0.8, 0.9, 1.0, 1.1, 1.2)
VOL_OF_VOL_SENSITIVITY_STRIKES = (0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5)
N_PATH_VALUES = tuple(1000 * 2**i for i in range(10))
BASE_SEED = 12345
THETA_SENSITIVITY_N_PATHS = 100_000
THETA_SENSITIVITY_VALUES = (0.04, 0.06, 0.08, 0.10, 0.12)
VOL_OF_VOL_SENSITIVITY_N_PATHS = 100_000
VOL_OF_VOL_SENSITIVITY_THETA = 0.12
VOL_OF_VOL_SENSITIVITY_VALUES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
VOL_OF_VOL_2_S0 = 2.0
VOL_OF_VOL_2_STRIKES = tuple(VOL_OF_VOL_2_S0 - 0.2 * 2 + i * 0.2 for i in range(5))
VOL_OF_VOL_3_S0 = 5.0
VOL_OF_VOL_3_STRIKES = tuple(VOL_OF_VOL_3_S0 - 0.5 * 2 + i * 0.5 for i in range(5))
KAPPA_SENSITIVITY_N_PATHS = 100_000
KAPPA_SENSITIVITY_S0 = VOL_OF_VOL_3_S0
KAPPA_SENSITIVITY_STRIKES = VOL_OF_VOL_3_STRIKES
KAPPA_SENSITIVITY_VOL_OF_VOL = 0.60
KAPPA_SENSITIVITY_VALUES = tuple(5 - 0.5 * i for i in range(7))
RHO_SENSITIVITY_N_PATHS = 100_000
RHO_SENSITIVITY_S0 = VOL_OF_VOL_3_S0
RHO_SENSITIVITY_STRIKES = VOL_OF_VOL_3_STRIKES
RHO_SENSITIVITY_VOL_OF_VOL = 0.60
RHO_SENSITIVITY_VALUES = tuple(0.9 - 0.3 * i for i in range(7))
CORRELATION_SENSITIVITY_N_PATHS = 100_000
CORRELATION_SENSITIVITY_S0 = 1.0
CORRELATION_SENSITIVITY_STRIKES = (0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5)
CORRELATION_SENSITIVITY_THETA = 0.12
CORRELATION_SENSITIVITY_KAPPA = 2.0
CORRELATION_SENSITIVITY_VOL_OF_VOL = 0.2
CORRELATION_SENSITIVITY_VALUES = (-0.9, -0.7, -0.5, -0.2, 0.0, 0.2, 0.5, 0.7, 0.9)
NEGATIVE_CORRELATION_VOL_OF_VOL = -0.7
CONFIDENCE_INTERVAL_MULTIPLIER = 1.96

GBM_SIGMA = sqrt(0.08)

HESTON_V0 = 0.08
HESTON_THETA = 0.08
HESTON_RHO = 0.3
HESTON_KAPPA = 2.0
HESTON_VOL_OF_VOL = 0.2

DEFAULT_OUTPUT_PATH = Path("output") / "lsmc_heston_sensitivities.xlsx"
SUMMARY_HEADERS = ["Strike", "GBM Estimated PV", "GBM Standard Error", "Heston Estimated PV", "Heston Standard Error"]
CONVERGENCE_HEADERS = ["Number of paths", "Strike", "Estimated PV", "Standard Error"]
AVERAGE_VARIANCE_HEADERS = [
    "Number of paths",
    "GBM Total Variance",
    "Average Heston Total Variance",
    "Standard Error",
    "95% CI Half Width",
]
PATH_VARIANCE_HEADERS = ["Path Number", "GBM Total Variance", "Heston Total Variance"]
THETA_SENSITIVITY_HEADERS = [
    "Strike",
    "GBM Estimated PV",
    "GBM Standard Error",
    "Theta",
    "Heston Estimated PV",
    "Heston Standard Error",
    "Average Heston Total Variance",
    "Variance Standard Error",
    "Variance 95% CI Half Width",
]
THETA_VARIANCE_HEADERS = ["Strike", "GBM"] + [
    f"Theta {theta:.2f}"
    for theta in THETA_SENSITIVITY_VALUES
]


def parameter_rows(
    path_counts: tuple[int, ...] | None = None,
    heston_theta: Any = HESTON_THETA,
    heston_vol_of_vol: Any = HESTON_VOL_OF_VOL,
    heston_kappa: Any = HESTON_KAPPA,
    heston_rho: Any = HESTON_RHO,
    s0: float = S0,
    strikes: tuple[float, ...] = STRIKES,
    maturity: float = MATURITY,
) -> list[list[Any]]:
    displayed_path_counts = N_PATH_VALUES if path_counts is None else path_counts
    return [
        ["Parameter", "Value"],
        ["S0", s0],
        ["Rate", RATE],
        ["Maturity", maturity],
        ["Steps per year", STEPS_PER_YEAR],
        ["Variance reduction", VARIANCE_REDUCTION_METHOD],
        ["Base seed", BASE_SEED],
        ["Strikes", ", ".join(str(strike) for strike in strikes)],
        ["Path counts", ", ".join(f"{n_paths:,}" for n_paths in displayed_path_counts)],
        ["GBM sigma", round(GBM_SIGMA, 8)],
        ["GBM total variance", round(GBM_SIGMA * GBM_SIGMA * MATURITY, 8)],
        ["Heston v0", HESTON_V0],
        ["Heston theta", heston_theta],
        ["Heston rho", heston_rho],
        ["Heston kappa", heston_kappa],
        ["Heston vol of vol", heston_vol_of_vol],
    ]


def format_float(value: float) -> str:
    return f"{0.0 if abs(value) < 1e-12 else value:.2f}"


def use_antithetic() -> bool:
    if VARIANCE_REDUCTION_METHOD not in VARIANCE_REDUCTION_METHODS:
        raise ValueError(f"VARIANCE_REDUCTION_METHOD must be one of {VARIANCE_REDUCTION_METHODS}")
    return VARIANCE_REDUCTION_METHOD == "antithetic"


def price_american_put_lsm(paths: Any, *, strike: float, rate: float, maturity: float) -> Any:
    if VARIANCE_REDUCTION_METHOD != "control_variate":
        return raw_price_american_put_lsm(paths, strike=strike, rate=rate, maturity=maturity)

    s0 = float(paths[0, 0])
    discount = np.exp(-rate * maturity)
    european_payoffs = discount * put_payoff(paths[-1], strike)
    european_price = european_put_price(s0, strike, rate, GBM_SIGMA, maturity)
    return raw_price_american_put_lsm(
        paths,
        strike=strike,
        rate=rate,
        maturity=maturity,
        control_variate_values=european_payoffs,
        control_variate_mean=european_price,
    )


def sheet_rows(
    headers: list[str],
    data_rows: list[list[Any]],
    path_counts: tuple[int, ...] | None = None,
    heston_theta: Any = HESTON_THETA,
    heston_vol_of_vol: Any = HESTON_VOL_OF_VOL,
    heston_kappa: Any = HESTON_KAPPA,
    heston_rho: Any = HESTON_RHO,
    s0: float = S0,
    strikes: tuple[float, ...] = STRIKES,
    maturity: float = MATURITY,
) -> list[list[Any]]:
    return parameter_rows(path_counts, heston_theta, heston_vol_of_vol, heston_kappa, heston_rho, s0, strikes, maturity) + [[]] + [headers] + data_rows


def first_data_row(
    path_counts: tuple[int, ...] | None = None,
    heston_theta: Any = HESTON_THETA,
    heston_vol_of_vol: Any = HESTON_VOL_OF_VOL,
    heston_kappa: Any = HESTON_KAPPA,
    heston_rho: Any = HESTON_RHO,
    s0: float = S0,
    strikes: tuple[float, ...] = STRIKES,
    maturity: float = MATURITY,
) -> int:
    return len(parameter_rows(path_counts, heston_theta, heston_vol_of_vol, heston_kappa, heston_rho, s0, strikes, maturity)) + 3


def calculate_model_results(model_name: str) -> list[list[Any]]:
    rows = []

    for path_index, n_paths in enumerate(N_PATH_VALUES):
        paths = simulate_model_paths(model_name, n_paths, BASE_SEED + path_index)
        for strike in STRIKES:
            result = price_american_put_lsm(
                paths,
                strike=strike,
                rate=RATE,
                maturity=MATURITY,
            )
            rows.append([n_paths, strike, round(result.price, 8), round(result.standard_error, 8)])

    return rows


def calculate_heston_results() -> tuple[list[list[Any]], list[list[Any]], list[list[Any]]]:
    rows = []
    average_variance_rows = []
    first_simulation_path_variance_rows = []
    smallest_n_paths = min(N_PATH_VALUES)

    for path_index, n_paths in enumerate(N_PATH_VALUES):
        spot_paths, variance_paths = simulate_heston_paths_with_variance(n_paths, BASE_SEED + path_index)
        total_variance_by_path = heston_total_variance_by_path(variance_paths)
        gbm_total_variance = GBM_SIGMA * GBM_SIGMA * MATURITY
        average_heston_variance, variance_standard_error, variance_ci_half_width = variance_summary(
            total_variance_by_path,
            n_paths,
        )
        average_variance_rows.append(
            [
                n_paths,
                round(gbm_total_variance, 8),
                average_heston_variance,
                variance_standard_error,
                variance_ci_half_width,
            ]
        )

        if n_paths == smallest_n_paths:
            gbm_total_variance = GBM_SIGMA * GBM_SIGMA * MATURITY
            first_simulation_path_variance_rows = [
                [path_number, round(gbm_total_variance, 8), round(float(total_variance), 8)]
                for path_number, total_variance in enumerate(total_variance_by_path, start=1)
            ]

        for strike in STRIKES:
            result = price_american_put_lsm(
                spot_paths,
                strike=strike,
                rate=RATE,
                maturity=MATURITY,
            )
            rows.append([n_paths, strike, round(result.price, 8), round(result.standard_error, 8)])

    return rows, average_variance_rows, first_simulation_path_variance_rows


def simulate_model_paths(model_name: str, n_paths: int, seed: int) -> Any:
    if model_name == "GBM":
        return simulate_gbm_paths(
            s0=S0,
            rate=RATE,
            sigma=GBM_SIGMA,
            maturity=MATURITY,
            steps_per_year=STEPS_PER_YEAR,
            n_paths=n_paths,
            seed=seed,
            antithetic=use_antithetic(),
        )
    if model_name == "Heston":
        spot_paths, _ = simulate_heston_paths_with_variance(n_paths, seed)
        return spot_paths
    raise ValueError(f"Unknown model: {model_name}")


def simulate_heston_paths_with_variance(n_paths: int, seed: int) -> tuple[Any, Any]:
    return simulate_heston_paths_with_variance_for_theta(n_paths, seed, HESTON_THETA)


def simulate_heston_paths_with_variance_for_theta(
    n_paths: int,
    seed: int,
    theta: float,
    vol_of_vol: float = HESTON_VOL_OF_VOL,
) -> tuple[Any, Any]:
    return simulate_heston_spot_scaled_vol_of_vol_paths(
        s0=S0,
        rate=RATE,
        v0=HESTON_V0,
        kappa=HESTON_KAPPA,
        theta=theta,
        vol_of_vol=vol_of_vol,
        rho=HESTON_RHO,
        maturity=MATURITY,
        steps_per_year=STEPS_PER_YEAR,
        n_paths=n_paths,
        seed=seed,
        antithetic=use_antithetic(),
        return_variance=True,
    )


def heston_total_variance_by_path(variance_paths: Any) -> Any:
    return total_variance_by_path(variance_paths, MATURITY)


def total_variance_by_path(variance_paths: Any, maturity: float) -> Any:
    dt = maturity / (variance_paths.shape[0] - 1)
    return np.sum(variance_paths[1:], axis=0) * dt


def variance_summary(total_variances_by_path: Any, n_paths: int) -> tuple[float, float, float]:
    standard_error = float(np.std(total_variances_by_path, ddof=1) / sqrt(n_paths))
    return (
        round(float(np.mean(total_variances_by_path)), 8),
        round(standard_error, 8),
        round(CONFIDENCE_INTERVAL_MULTIPLIER * standard_error, 8),
    )


def confidence_interval_half_width(standard_error: float) -> float:
    return round(CONFIDENCE_INTERVAL_MULTIPLIER * standard_error, 8)


def build_summary_sheet(sheet_name: str, gbm_rows: list[list[Any]], heston_rows: list[list[Any]]) -> Sheet:
    largest_n_paths = max(N_PATH_VALUES)
    data_rows = [
        [
            strike,
            _find_result(gbm_rows, largest_n_paths, strike)[2],
            _find_result(gbm_rows, largest_n_paths, strike)[3],
            _find_result(heston_rows, largest_n_paths, strike)[2],
            _find_result(heston_rows, largest_n_paths, strike)[3],
        ]
        for strike in STRIKES
    ]
    summary_path_counts = (largest_n_paths,)
    data_start = first_data_row(summary_path_counts)
    data_end = data_start + len(data_rows) - 1
    chart: ChartSpec = (
        f"GBM versus Heston estimated PV by strike, {largest_n_paths:,} paths",
        1,
        8,
        [
            (
                "GBM",
                f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
            ),
            (
                "Heston",
                f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                f"'{sheet_name}'!$E${data_start}:$E${data_end}",
            ),
        ],
    )
    return (sheet_name, [], sheet_rows(SUMMARY_HEADERS, data_rows, summary_path_counts), [chart])


def _find_result(rows: list[list[Any]], n_paths: int, strike: float) -> list[Any]:
    return next(row for row in rows if row[0] == n_paths and row[1] == strike)


def build_convergence_sheet(sheet_name: str, convergence_rows: list[list[Any]]) -> Sheet:
    data_rows = sorted(convergence_rows, key=lambda row: (row[1], row[0]))
    charts = []
    data_start = first_data_row()

    for strike_index, strike in enumerate(STRIKES):
        first_row = data_start + strike_index * len(N_PATH_VALUES)
        last_row = first_row + len(N_PATH_VALUES) - 1
        charts.append(
            (
                f"{sheet_name} convergence, strike={strike:.1f}",
                1 + strike_index * 16,
                7,
                [
                    (
                        f"Strike {strike:.1f}",
                        f"'{sheet_name}'!$A${first_row}:$A${last_row}",
                        f"'{sheet_name}'!$C${first_row}:$C${last_row}",
                        f"'{sheet_name}'!$D${first_row}:$D${last_row}",
                    )
                ],
            )
        )

    return (sheet_name, [], sheet_rows(CONVERGENCE_HEADERS, data_rows), charts)


def build_average_variance_sheet(rows: list[list[Any]]) -> Sheet:
    sheet_name = "Base Heston Average Variance"
    data_start = first_data_row()
    data_end = data_start + len(rows) - 1
    chart: ChartSpec = (
        "Average Heston total variance by number of paths",
        1,
        5,
        [
            (
                "GBM total variance",
                f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                "",
            ),
            (
                "Average Heston total variance",
                f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                f"'{sheet_name}'!$E${data_start}:$E${data_end}",
            )
        ],
    )
    return (sheet_name, [], sheet_rows(AVERAGE_VARIANCE_HEADERS, rows), [chart])


def build_path_variance_sheet(rows: list[list[Any]]) -> Sheet:
    smallest_n_paths = min(N_PATH_VALUES)
    return (f"Base Path Variance {smallest_n_paths:,}", [], sheet_rows(PATH_VARIANCE_HEADERS, rows))


def calculate_theta_sensitivity_rows() -> list[list[Any]]:
    rows = []
    gbm_paths = simulate_model_paths("GBM", THETA_SENSITIVITY_N_PATHS, BASE_SEED + 10_000)
    gbm_results = {}
    for strike in STRIKES:
        result = price_american_put_lsm(
            gbm_paths,
            strike=strike,
            rate=RATE,
            maturity=MATURITY,
        )
        gbm_results[strike] = (round(result.price, 8), round(result.standard_error, 8))

    for theta_index, theta in enumerate(THETA_SENSITIVITY_VALUES):
        heston_paths, variance_paths = simulate_heston_paths_with_variance_for_theta(
            THETA_SENSITIVITY_N_PATHS,
            BASE_SEED + 20_000 + theta_index,
            theta,
        )
        average_heston_variance, variance_standard_error, variance_ci_half_width = variance_summary(
            heston_total_variance_by_path(variance_paths),
            THETA_SENSITIVITY_N_PATHS,
        )
        for strike in STRIKES:
            result = price_american_put_lsm(
                heston_paths,
                strike=strike,
                rate=RATE,
                maturity=MATURITY,
            )
            rows.append(
                [
                    strike,
                    gbm_results[strike][0],
                    gbm_results[strike][1],
                    theta,
                    round(result.price, 8),
                    round(result.standard_error, 8),
                    average_heston_variance,
                    variance_standard_error,
                    variance_ci_half_width,
                ]
            )

    return rows


def build_theta_sensitivity_sheet(rows: list[list[Any]]) -> Sheet:
    sheet_name = "Heston - change Theta"
    data_rows = []
    for strike in STRIKES:
        row = [strike]
        gbm_row = next(item for item in rows if item[0] == strike)
        row.extend([gbm_row[1], gbm_row[2]])
        for theta in THETA_SENSITIVITY_VALUES:
            heston_row = next(item for item in rows if item[0] == strike and item[3] == theta)
            row.extend([heston_row[4], heston_row[5]])
        data_rows.append(row)

    headers = ["Strike", "GBM Estimated PV", "GBM Standard Error"]
    for theta in THETA_SENSITIVITY_VALUES:
        headers.extend([f"Heston Theta {theta:.2f} Estimated PV", f"Heston Theta {theta:.2f} Standard Error"])

    gbm_total_variance = round(GBM_SIGMA * GBM_SIGMA * MATURITY, 8)
    variance_rows = []
    for strike in STRIKES:
        row = [strike, gbm_total_variance]
        for theta in THETA_SENSITIVITY_VALUES:
            heston_row = next(item for item in rows if item[0] == strike and item[3] == theta)
            row.append(heston_row[6])
        variance_rows.append(row)
    variance_chart_headers = [
        "Theta",
        "GBM Variance",
        "Heston Average Variance",
        "Heston Variance Standard Error",
        "Heston Variance 95% CI Half Width",
    ]
    variance_chart_rows = [
        [
            f"Theta {theta:.2f}",
            gbm_total_variance,
            next(item for item in rows if item[3] == theta)[6],
            next(item for item in rows if item[3] == theta)[7],
            next(item for item in rows if item[3] == theta)[8],
        ]
        for theta in THETA_SENSITIVITY_VALUES
    ]

    theta_path_counts = (THETA_SENSITIVITY_N_PATHS,)
    theta_display = ", ".join(f"{theta:.2f}" for theta in THETA_SENSITIVITY_VALUES)
    data_start = first_data_row(theta_path_counts, heston_theta=theta_display)
    data_end = data_start + len(data_rows) - 1
    series = [
        (
            "GBM",
            f"'{sheet_name}'!$A${data_start}:$A${data_end}",
            f"'{sheet_name}'!$B${data_start}:$B${data_end}",
            f"'{sheet_name}'!$C${data_start}:$C${data_end}",
        )
    ]
    for index, theta in enumerate(THETA_SENSITIVITY_VALUES):
        value_column = _column_name(4 + index * 2)
        error_column = _column_name(5 + index * 2)
        series.append(
            (
                f"Heston theta {theta:.2f}",
                f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                f"'{sheet_name}'!${value_column}${data_start}:${value_column}${data_end}",
                f"'{sheet_name}'!${error_column}${data_start}:${error_column}${data_end}",
            )
        )

    price_chart: ChartSpec = (
        f"GBM and Heston prices by theta, {THETA_SENSITIVITY_N_PATHS:,} paths",
        1,
        16,
        series,
    )
    worksheet_rows = (
        parameter_rows(theta_path_counts, heston_theta=theta_display)
        + [[]]
        + [headers]
        + data_rows
        + [[]]
        + [THETA_VARIANCE_HEADERS]
        + variance_rows
        + [[]]
        + [variance_chart_headers]
        + variance_chart_rows
    )
    variance_chart_header_row = data_end + 2 + 1 + len(variance_rows) + 1
    variance_chart_data_start = variance_chart_header_row + 1
    variance_chart_data_end = variance_chart_data_start + len(variance_chart_rows) - 1
    variance_chart_series = [
        (
            "GBM variance",
            f"'{sheet_name}'!$A${variance_chart_data_start}:$A${variance_chart_data_end}",
            f"'{sheet_name}'!$B${variance_chart_data_start}:$B${variance_chart_data_end}",
            "",
        ),
        (
            "Heston average variance",
            f"'{sheet_name}'!$A${variance_chart_data_start}:$A${variance_chart_data_end}",
            f"'{sheet_name}'!$C${variance_chart_data_start}:$C${variance_chart_data_end}",
            f"'{sheet_name}'!$E${variance_chart_data_start}:$E${variance_chart_data_end}",
        ),
    ]
    variance_chart: ChartSpec = (
        f"GBM and Heston average variance by theta, {THETA_SENSITIVITY_N_PATHS:,} paths",
        17,
        16,
        variance_chart_series,
    )
    return (sheet_name, [], worksheet_rows, [price_chart, variance_chart])


def calculate_vol_of_vol_sensitivity_rows() -> list[list[Any]]:
    return calculate_vol_of_vol_sensitivity_rows_for_scenario(
        S0,
        VOL_OF_VOL_SENSITIVITY_STRIKES,
        MATURITY,
        BASE_SEED + 30_000,
        BASE_SEED + 40_000,
        HESTON_RHO,
    )


def calculate_negative_correlation_vol_of_vol_sensitivity_rows() -> list[list[Any]]:
    return calculate_vol_of_vol_sensitivity_rows_for_scenario(
        S0,
        VOL_OF_VOL_SENSITIVITY_STRIKES,
        MATURITY,
        BASE_SEED + 90_000,
        BASE_SEED + 100_000,
        NEGATIVE_CORRELATION_VOL_OF_VOL,
    )


def calculate_vol_of_vol_2_sensitivity_rows() -> list[list[Any]]:
    return calculate_vol_of_vol_sensitivity_rows_for_scenario(
        VOL_OF_VOL_2_S0,
        VOL_OF_VOL_2_STRIKES,
        MATURITY,
        BASE_SEED + 50_000,
        BASE_SEED + 60_000,
    )


def calculate_vol_of_vol_3_sensitivity_rows() -> list[list[Any]]:
    return calculate_vol_of_vol_sensitivity_rows_for_scenario(
        VOL_OF_VOL_3_S0,
        VOL_OF_VOL_3_STRIKES,
        MATURITY,
        BASE_SEED + 70_000,
        BASE_SEED + 80_000,
    )


def calculate_vol_of_vol_sensitivity_rows_for_scenario(
    s0: float,
    strikes: tuple[float, ...],
    maturity: float,
    gbm_seed: int,
    heston_seed_base: int,
    rho: float = HESTON_RHO,
) -> list[list[Any]]:
    rows = []
    gbm_paths = simulate_gbm_paths(
        s0=s0,
        rate=RATE,
        sigma=GBM_SIGMA,
        maturity=maturity,
        steps_per_year=STEPS_PER_YEAR,
        n_paths=VOL_OF_VOL_SENSITIVITY_N_PATHS,
        seed=gbm_seed,
        antithetic=use_antithetic(),
    )
    gbm_results = {}
    for strike in strikes:
        result = price_american_put_lsm(
            gbm_paths,
            strike=strike,
            rate=RATE,
            maturity=maturity,
        )
        gbm_results[strike] = (round(result.price, 8), round(result.standard_error, 8))

    for vol_index, vol_of_vol in enumerate(VOL_OF_VOL_SENSITIVITY_VALUES):
        heston_paths, variance_paths = simulate_heston_spot_scaled_vol_of_vol_paths(
            s0=s0,
            rate=RATE,
            v0=HESTON_V0,
            kappa=HESTON_KAPPA,
            theta=VOL_OF_VOL_SENSITIVITY_THETA,
            vol_of_vol=vol_of_vol,
            rho=rho,
            maturity=maturity,
            steps_per_year=STEPS_PER_YEAR,
            n_paths=VOL_OF_VOL_SENSITIVITY_N_PATHS,
            seed=heston_seed_base + vol_index,
            antithetic=use_antithetic(),
            return_variance=True,
        )
        average_heston_variance, variance_standard_error, variance_ci_half_width = variance_summary(
            total_variance_by_path(variance_paths, maturity),
            VOL_OF_VOL_SENSITIVITY_N_PATHS,
        )
        for strike in strikes:
            result = price_american_put_lsm(
                heston_paths,
                strike=strike,
                rate=RATE,
                maturity=maturity,
            )
            rows.append(
                [
                    strike,
                    gbm_results[strike][0],
                    gbm_results[strike][1],
                    vol_of_vol,
                    round(result.price, 8),
                    round(result.standard_error, 8),
                    average_heston_variance,
                    variance_standard_error,
                    variance_ci_half_width,
                ]
            )

    return rows


def build_vol_of_vol_sensitivity_sheet(rows: list[list[Any]]) -> Sheet:
    return build_vol_of_vol_sheet(
        "Heston - chg Vol_vol corr=0.3",
        rows,
        S0,
        VOL_OF_VOL_SENSITIVITY_STRIKES,
        MATURITY,
        HESTON_RHO,
    )


def build_negative_correlation_vol_of_vol_sensitivity_sheet(rows: list[list[Any]]) -> Sheet:
    return build_vol_of_vol_sheet(
        "Heston - cng Vol_vol corr=-0.7",
        rows,
        S0,
        VOL_OF_VOL_SENSITIVITY_STRIKES,
        MATURITY,
        NEGATIVE_CORRELATION_VOL_OF_VOL,
    )


def build_vol_of_vol_2_sensitivity_sheet(rows: list[list[Any]]) -> Sheet:
    return build_vol_of_vol_sheet(
        "Heston - change Vol_vol 2",
        rows,
        VOL_OF_VOL_2_S0,
        VOL_OF_VOL_2_STRIKES,
        MATURITY,
        HESTON_RHO,
    )


def build_vol_of_vol_3_sensitivity_sheet(rows: list[list[Any]]) -> Sheet:
    return build_vol_of_vol_sheet(
        "Heston - change Vol_vol 3",
        rows,
        VOL_OF_VOL_3_S0,
        VOL_OF_VOL_3_STRIKES,
        MATURITY,
        HESTON_RHO,
    )


def build_vol_of_vol_sheet(
    sheet_name: str,
    rows: list[list[Any]],
    s0: float,
    strikes: tuple[float, ...],
    maturity: float,
    heston_rho: float,
) -> Sheet:
    data_rows = []
    for strike in strikes:
        row = [strike]
        gbm_row = next(item for item in rows if item[0] == strike)
        row.extend([gbm_row[1], gbm_row[2]])
        for vol_of_vol in VOL_OF_VOL_SENSITIVITY_VALUES:
            heston_row = next(item for item in rows if item[0] == strike and item[3] == vol_of_vol)
            row.extend([heston_row[4], heston_row[5]])
        data_rows.append(row)

    headers = ["Strike", "GBM Estimated PV", "GBM Standard Error"]
    for vol_of_vol in VOL_OF_VOL_SENSITIVITY_VALUES:
        headers.extend(
            [
                f"Heston Vol_vol {vol_of_vol:.2f} Estimated PV",
                f"Heston Vol_vol {vol_of_vol:.2f} Standard Error",
            ]
        )

    gbm_total_variance = round(GBM_SIGMA * GBM_SIGMA * maturity, 8)
    variance_headers = ["Strike", "GBM"] + [
        f"Vol_vol {vol_of_vol:.2f}"
        for vol_of_vol in VOL_OF_VOL_SENSITIVITY_VALUES
    ]
    variance_rows = []
    for strike in strikes:
        row = [strike, gbm_total_variance]
        for vol_of_vol in VOL_OF_VOL_SENSITIVITY_VALUES:
            heston_row = next(item for item in rows if item[0] == strike and item[3] == vol_of_vol)
            row.append(heston_row[6])
        variance_rows.append(row)

    variance_chart_headers = [
        "Vol_vol",
        "GBM Variance",
        "Heston Average Variance",
        "Heston Variance Standard Error",
        "Heston Variance 95% CI Half Width",
    ]
    variance_chart_rows = [
        [
            f"Vol_vol {vol_of_vol:.2f}",
            gbm_total_variance,
            next(item for item in rows if item[3] == vol_of_vol)[6],
            next(item for item in rows if item[3] == vol_of_vol)[7],
            next(item for item in rows if item[3] == vol_of_vol)[8],
        ]
        for vol_of_vol in VOL_OF_VOL_SENSITIVITY_VALUES
    ]

    strike_price_headers = ["Model / Vol_vol", "Estimated PV", "Standard Error", "95% CI Half Width"]
    strike_price_sections = []
    for strike in strikes:
        gbm_row = next(item for item in rows if item[0] == strike)
        section_rows = [
            [
                "GBM",
                gbm_row[1],
                gbm_row[2],
                confidence_interval_half_width(gbm_row[2]),
            ]
        ]
        for vol_of_vol in VOL_OF_VOL_SENSITIVITY_VALUES:
            heston_row = next(item for item in rows if item[0] == strike and item[3] == vol_of_vol)
            section_rows.append(
                [
                    f"Heston vol_vol {vol_of_vol:.2f}",
                    heston_row[4],
                    heston_row[5],
                    confidence_interval_half_width(heston_row[5]),
                ]
            )
        strike_price_sections.append((strike, section_rows))

    path_counts = (VOL_OF_VOL_SENSITIVITY_N_PATHS,)
    vol_of_vol_display = ", ".join(f"{vol_of_vol:.2f}" for vol_of_vol in VOL_OF_VOL_SENSITIVITY_VALUES)
    data_start = first_data_row(
        path_counts,
        heston_theta=VOL_OF_VOL_SENSITIVITY_THETA,
        heston_vol_of_vol=vol_of_vol_display,
        s0=s0,
        strikes=strikes,
        maturity=maturity,
    )
    data_end = data_start + len(data_rows) - 1
    series = [
        (
            "GBM",
            f"'{sheet_name}'!$A${data_start}:$A${data_end}",
            f"'{sheet_name}'!$B${data_start}:$B${data_end}",
            f"'{sheet_name}'!$C${data_start}:$C${data_end}",
        )
    ]
    for index, vol_of_vol in enumerate(VOL_OF_VOL_SENSITIVITY_VALUES):
        value_column = _column_name(4 + index * 2)
        error_column = _column_name(5 + index * 2)
        series.append(
            (
                f"Heston vol_vol {vol_of_vol:.2f}",
                f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                f"'{sheet_name}'!${value_column}${data_start}:${value_column}${data_end}",
                f"'{sheet_name}'!${error_column}${data_start}:${error_column}${data_end}",
            )
        )

    price_chart: ChartSpec = (
        f"GBM and Heston prices by vol_of_vol, {VOL_OF_VOL_SENSITIVITY_N_PATHS:,} paths",
        1,
        16,
        series,
    )
    worksheet_rows = (
        parameter_rows(
            path_counts,
            heston_theta=VOL_OF_VOL_SENSITIVITY_THETA,
            heston_vol_of_vol=vol_of_vol_display,
            heston_rho=heston_rho,
            s0=s0,
            strikes=strikes,
            maturity=maturity,
        )
        + [[]]
        + [headers]
        + data_rows
        + [[]]
        + [variance_headers]
        + variance_rows
        + [[]]
        + [variance_chart_headers]
        + variance_chart_rows
    )
    strike_price_section_ranges = []
    for strike, strike_price_rows in strike_price_sections:
        worksheet_rows.append([])
        worksheet_rows.append([f"Price by model and vol_of_vol for strike {strike:.2f}"])
        worksheet_rows.append(strike_price_headers)
        section_data_start = len(worksheet_rows) + 1
        worksheet_rows.extend(strike_price_rows)
        section_data_end = section_data_start + len(strike_price_rows) - 1
        strike_price_section_ranges.append((strike, section_data_start, section_data_end))

    variance_chart_header_row = data_end + 2 + 1 + len(variance_rows) + 1
    variance_chart_data_start = variance_chart_header_row + 1
    variance_chart_data_end = variance_chart_data_start + len(variance_chart_rows) - 1
    variance_chart: ChartSpec = (
        f"GBM and Heston average variance by vol_of_vol, {VOL_OF_VOL_SENSITIVITY_N_PATHS:,} paths",
        17,
        16,
        [
            (
                "GBM variance",
                f"'{sheet_name}'!$A${variance_chart_data_start}:$A${variance_chart_data_end}",
                f"'{sheet_name}'!$B${variance_chart_data_start}:$B${variance_chart_data_end}",
                "",
            ),
            (
                "Heston average variance",
                f"'{sheet_name}'!$A${variance_chart_data_start}:$A${variance_chart_data_end}",
                f"'{sheet_name}'!$C${variance_chart_data_start}:$C${variance_chart_data_end}",
                f"'{sheet_name}'!$E${variance_chart_data_start}:$E${variance_chart_data_end}",
            ),
        ],
    )
    strike_price_charts: list[ChartSpec] = []
    for strike_index, (strike, section_data_start, section_data_end) in enumerate(strike_price_section_ranges):
        strike_price_charts.append(
            (
                f"Price by GBM and Heston vol_of_vol, strike={strike:.2f}",
                33 + strike_index * 16,
                16,
                [
                    (
                        f"Strike {strike:.2f}",
                        f"'{sheet_name}'!$A${section_data_start}:$A${section_data_end}",
                        f"'{sheet_name}'!$B${section_data_start}:$B${section_data_end}",
                        f"'{sheet_name}'!$D${section_data_start}:$D${section_data_end}",
                    )
                ],
            )
        )

    return (sheet_name, [], worksheet_rows, [price_chart, variance_chart] + strike_price_charts)


def calculate_kappa_sensitivity_rows() -> list[list[Any]]:
    return calculate_kappa_sensitivity_rows_for_scenario(
        KAPPA_SENSITIVITY_S0,
        KAPPA_SENSITIVITY_STRIKES,
        MATURITY,
        BASE_SEED + 90_000,
        BASE_SEED + 100_000,
        VOL_OF_VOL_SENSITIVITY_THETA,
        KAPPA_SENSITIVITY_VOL_OF_VOL,
        HESTON_RHO,
    )


def calculate_kappa_sensitivity_rows_for_scenario(
    s0: float,
    strikes: tuple[float, ...],
    maturity: float,
    gbm_seed: int,
    heston_seed_base: int,
    theta: float,
    vol_of_vol: float,
    rho: float,
) -> list[list[Any]]:
    rows = []
    gbm_paths = simulate_gbm_paths(
        s0=s0,
        rate=RATE,
        sigma=GBM_SIGMA,
        maturity=maturity,
        steps_per_year=STEPS_PER_YEAR,
        n_paths=KAPPA_SENSITIVITY_N_PATHS,
        seed=gbm_seed,
        antithetic=use_antithetic(),
    )
    gbm_results = {}
    for strike in strikes:
        result = price_american_put_lsm(
            gbm_paths,
            strike=strike,
            rate=RATE,
            maturity=maturity,
        )
        gbm_results[strike] = (round(result.price, 8), round(result.standard_error, 8))

    for kappa_index, kappa in enumerate(KAPPA_SENSITIVITY_VALUES):
        heston_paths, variance_paths = simulate_heston_spot_scaled_vol_of_vol_paths(
            s0=s0,
            rate=RATE,
            v0=HESTON_V0,
            kappa=kappa,
            theta=theta,
            vol_of_vol=vol_of_vol,
            rho=rho,
            maturity=maturity,
            steps_per_year=STEPS_PER_YEAR,
            n_paths=KAPPA_SENSITIVITY_N_PATHS,
            seed=heston_seed_base + kappa_index,
            antithetic=use_antithetic(),
            return_variance=True,
        )
        average_heston_variance, variance_standard_error, variance_ci_half_width = variance_summary(
            total_variance_by_path(variance_paths, maturity),
            KAPPA_SENSITIVITY_N_PATHS,
        )
        for strike in strikes:
            result = price_american_put_lsm(
                heston_paths,
                strike=strike,
                rate=RATE,
                maturity=maturity,
            )
            rows.append(
                [
                    strike,
                    gbm_results[strike][0],
                    gbm_results[strike][1],
                    kappa,
                    round(result.price, 8),
                    round(result.standard_error, 8),
                    average_heston_variance,
                    variance_standard_error,
                    variance_ci_half_width,
                ]
            )

    return rows


def build_kappa_sensitivity_sheet(rows: list[list[Any]]) -> Sheet:
    return build_kappa_sheet(
        "Heston - change Kappa",
        rows,
        KAPPA_SENSITIVITY_S0,
        KAPPA_SENSITIVITY_STRIKES,
        MATURITY,
        VOL_OF_VOL_SENSITIVITY_THETA,
        KAPPA_SENSITIVITY_VOL_OF_VOL,
        HESTON_RHO,
    )


def calculate_kappa_sensitivity_rows_corr_0_3() -> list[list[Any]]:
    return calculate_kappa_sensitivity_rows_for_scenario(
        S0,
        VOL_OF_VOL_SENSITIVITY_STRIKES,
        MATURITY,
        BASE_SEED + 130_000,
        BASE_SEED + 140_000,
        VOL_OF_VOL_SENSITIVITY_THETA,
        HESTON_VOL_OF_VOL,
        HESTON_RHO,
    )


def calculate_kappa_sensitivity_rows_corr_negative_0_7() -> list[list[Any]]:
    return calculate_kappa_sensitivity_rows_for_scenario(
        S0,
        VOL_OF_VOL_SENSITIVITY_STRIKES,
        MATURITY,
        BASE_SEED + 150_000,
        BASE_SEED + 160_000,
        VOL_OF_VOL_SENSITIVITY_THETA,
        HESTON_VOL_OF_VOL,
        NEGATIVE_CORRELATION_VOL_OF_VOL,
    )


def build_kappa_sensitivity_sheet_corr_0_3(rows: list[list[Any]]) -> Sheet:
    return build_kappa_sheet(
        "Heston - cng Kappa corr=0.3",
        rows,
        S0,
        VOL_OF_VOL_SENSITIVITY_STRIKES,
        MATURITY,
        VOL_OF_VOL_SENSITIVITY_THETA,
        HESTON_VOL_OF_VOL,
        HESTON_RHO,
    )


def build_kappa_sensitivity_sheet_corr_negative_0_7(rows: list[list[Any]]) -> Sheet:
    return build_kappa_sheet(
        "Heston - cng Kappa corr=-0.7",
        rows,
        S0,
        VOL_OF_VOL_SENSITIVITY_STRIKES,
        MATURITY,
        VOL_OF_VOL_SENSITIVITY_THETA,
        HESTON_VOL_OF_VOL,
        NEGATIVE_CORRELATION_VOL_OF_VOL,
    )


def build_kappa_sheet(
    sheet_name: str,
    rows: list[list[Any]],
    s0: float,
    strikes: tuple[float, ...],
    maturity: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
) -> Sheet:
    data_rows = []
    for strike in strikes:
        row = [strike]
        gbm_row = next(item for item in rows if item[0] == strike)
        row.extend([gbm_row[1], gbm_row[2]])
        for kappa in KAPPA_SENSITIVITY_VALUES:
            heston_row = next(item for item in rows if item[0] == strike and item[3] == kappa)
            row.extend([heston_row[4], heston_row[5]])
        data_rows.append(row)

    headers = ["Strike", "GBM Estimated PV", "GBM Standard Error"]
    for kappa in KAPPA_SENSITIVITY_VALUES:
        headers.extend([f"Heston Kappa {kappa:.2f} Estimated PV", f"Heston Kappa {kappa:.2f} Standard Error"])

    gbm_total_variance = round(GBM_SIGMA * GBM_SIGMA * maturity, 8)
    variance_headers = ["Strike", "GBM"] + [
        f"Kappa {kappa:.2f}"
        for kappa in KAPPA_SENSITIVITY_VALUES
    ]
    variance_rows = []
    for strike in strikes:
        row = [strike, gbm_total_variance]
        for kappa in KAPPA_SENSITIVITY_VALUES:
            heston_row = next(item for item in rows if item[0] == strike and item[3] == kappa)
            row.append(heston_row[6])
        variance_rows.append(row)

    variance_chart_headers = [
        "Kappa",
        "GBM Variance",
        "Heston Average Variance",
        "Heston Variance Standard Error",
        "Heston Variance 95% CI Half Width",
    ]
    variance_chart_rows = [
        [
            f"Kappa {kappa:.2f}",
            gbm_total_variance,
            next(item for item in rows if item[3] == kappa)[6],
            next(item for item in rows if item[3] == kappa)[7],
            next(item for item in rows if item[3] == kappa)[8],
        ]
        for kappa in KAPPA_SENSITIVITY_VALUES
    ]

    strike_price_headers = ["Model / Kappa", "Estimated PV", "Standard Error", "95% CI Half Width"]
    strike_price_sections = []
    for strike in strikes:
        gbm_row = next(item for item in rows if item[0] == strike)
        section_rows = [
            [
                "GBM",
                gbm_row[1],
                gbm_row[2],
                confidence_interval_half_width(gbm_row[2]),
            ]
        ]
        for kappa in KAPPA_SENSITIVITY_VALUES:
            heston_row = next(item for item in rows if item[0] == strike and item[3] == kappa)
            section_rows.append(
                [
                    f"Heston kappa {kappa:.2f}",
                    heston_row[4],
                    heston_row[5],
                    confidence_interval_half_width(heston_row[5]),
                ]
            )
        strike_price_sections.append((strike, section_rows))

    path_counts = (KAPPA_SENSITIVITY_N_PATHS,)
    kappa_display = ", ".join(f"{kappa:.2f}" for kappa in KAPPA_SENSITIVITY_VALUES)
    data_start = first_data_row(
        path_counts,
        heston_theta=theta,
        heston_vol_of_vol=vol_of_vol,
        heston_kappa=kappa_display,
        heston_rho=rho,
        s0=s0,
        strikes=strikes,
        maturity=maturity,
    )
    data_end = data_start + len(data_rows) - 1
    series = [
        (
            "GBM",
            f"'{sheet_name}'!$A${data_start}:$A${data_end}",
            f"'{sheet_name}'!$B${data_start}:$B${data_end}",
            f"'{sheet_name}'!$C${data_start}:$C${data_end}",
        )
    ]
    for index, kappa in enumerate(KAPPA_SENSITIVITY_VALUES):
        value_column = _column_name(4 + index * 2)
        error_column = _column_name(5 + index * 2)
        series.append(
            (
                f"Heston kappa {kappa:.2f}",
                f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                f"'{sheet_name}'!${value_column}${data_start}:${value_column}${data_end}",
                f"'{sheet_name}'!${error_column}${data_start}:${error_column}${data_end}",
            )
        )

    price_chart: ChartSpec = (
        f"GBM and Heston prices by kappa, {KAPPA_SENSITIVITY_N_PATHS:,} paths",
        1,
        18,
        series,
    )
    worksheet_rows = (
        parameter_rows(
            path_counts,
            heston_theta=theta,
            heston_vol_of_vol=vol_of_vol,
            heston_kappa=kappa_display,
            heston_rho=rho,
            s0=s0,
            strikes=strikes,
            maturity=maturity,
        )
        + [[]]
        + [headers]
        + data_rows
        + [[]]
        + [variance_headers]
        + variance_rows
        + [[]]
        + [variance_chart_headers]
        + variance_chart_rows
    )
    strike_price_section_ranges = []
    for strike, strike_price_rows in strike_price_sections:
        worksheet_rows.append([])
        worksheet_rows.append([f"Price by model and kappa for strike {strike:.2f}"])
        worksheet_rows.append(strike_price_headers)
        section_data_start = len(worksheet_rows) + 1
        worksheet_rows.extend(strike_price_rows)
        section_data_end = section_data_start + len(strike_price_rows) - 1
        strike_price_section_ranges.append((strike, section_data_start, section_data_end))

    variance_chart_header_row = data_end + 2 + 1 + len(variance_rows) + 1
    variance_chart_data_start = variance_chart_header_row + 1
    variance_chart_data_end = variance_chart_data_start + len(variance_chart_rows) - 1
    variance_chart: ChartSpec = (
        f"GBM and Heston average variance by kappa, {KAPPA_SENSITIVITY_N_PATHS:,} paths",
        17,
        18,
        [
            (
                "GBM variance",
                f"'{sheet_name}'!$A${variance_chart_data_start}:$A${variance_chart_data_end}",
                f"'{sheet_name}'!$B${variance_chart_data_start}:$B${variance_chart_data_end}",
                "",
            ),
            (
                "Heston average variance",
                f"'{sheet_name}'!$A${variance_chart_data_start}:$A${variance_chart_data_end}",
                f"'{sheet_name}'!$C${variance_chart_data_start}:$C${variance_chart_data_end}",
                f"'{sheet_name}'!$E${variance_chart_data_start}:$E${variance_chart_data_end}",
            ),
        ],
    )
    strike_price_charts: list[ChartSpec] = []
    for strike_index, (strike, section_data_start, section_data_end) in enumerate(strike_price_section_ranges):
        strike_price_charts.append(
            (
                f"Price by GBM and Heston kappa, strike={strike:.2f}",
                33 + strike_index * 16,
                18,
                [
                    (
                        f"Strike {strike:.2f}",
                        f"'{sheet_name}'!$A${section_data_start}:$A${section_data_end}",
                        f"'{sheet_name}'!$B${section_data_start}:$B${section_data_end}",
                        f"'{sheet_name}'!$D${section_data_start}:$D${section_data_end}",
                    )
                ],
            )
        )

    return (sheet_name, [], worksheet_rows, [price_chart, variance_chart] + strike_price_charts)


def calculate_rho_sensitivity_rows() -> list[list[Any]]:
    return calculate_correlation_sensitivity_rows_for_scenario(
        RHO_SENSITIVITY_S0,
        RHO_SENSITIVITY_STRIKES,
        MATURITY,
        BASE_SEED + 110_000,
        BASE_SEED + 120_000,
        VOL_OF_VOL_SENSITIVITY_THETA,
        HESTON_KAPPA,
        RHO_SENSITIVITY_VOL_OF_VOL,
        RHO_SENSITIVITY_VALUES,
        RHO_SENSITIVITY_N_PATHS,
    )


def calculate_correlation_sensitivity_rows() -> list[list[Any]]:
    return calculate_correlation_sensitivity_rows_for_scenario(
        CORRELATION_SENSITIVITY_S0,
        CORRELATION_SENSITIVITY_STRIKES,
        MATURITY,
        BASE_SEED + 170_000,
        BASE_SEED + 180_000,
        CORRELATION_SENSITIVITY_THETA,
        CORRELATION_SENSITIVITY_KAPPA,
        CORRELATION_SENSITIVITY_VOL_OF_VOL,
        CORRELATION_SENSITIVITY_VALUES,
        CORRELATION_SENSITIVITY_N_PATHS,
    )


def calculate_correlation_sensitivity_rows_for_scenario(
    s0: float,
    strikes: tuple[float, ...],
    maturity: float,
    gbm_seed: int,
    heston_seed_base: int,
    theta: float,
    kappa: float,
    vol_of_vol: float,
    rho_values: tuple[float, ...],
    n_paths: int,
) -> list[list[Any]]:
    rows = []
    gbm_paths = simulate_gbm_paths(
        s0=s0,
        rate=RATE,
        sigma=GBM_SIGMA,
        maturity=maturity,
        steps_per_year=STEPS_PER_YEAR,
        n_paths=n_paths,
        seed=gbm_seed,
        antithetic=use_antithetic(),
    )
    gbm_results = {}
    for strike in strikes:
        result = price_american_put_lsm(
            gbm_paths,
            strike=strike,
            rate=RATE,
            maturity=maturity,
        )
        gbm_results[strike] = (round(result.price, 8), round(result.standard_error, 8))

    for rho_index, rho in enumerate(rho_values):
        heston_paths, variance_paths = simulate_heston_spot_scaled_vol_of_vol_paths(
            s0=s0,
            rate=RATE,
            v0=HESTON_V0,
            kappa=kappa,
            theta=theta,
            vol_of_vol=vol_of_vol,
            rho=rho,
            maturity=maturity,
            steps_per_year=STEPS_PER_YEAR,
            n_paths=n_paths,
            seed=heston_seed_base + rho_index,
            antithetic=use_antithetic(),
            return_variance=True,
        )
        average_heston_variance, variance_standard_error, variance_ci_half_width = variance_summary(
            total_variance_by_path(variance_paths, maturity),
            n_paths,
        )
        for strike in strikes:
            result = price_american_put_lsm(
                heston_paths,
                strike=strike,
                rate=RATE,
                maturity=maturity,
            )
            rows.append(
                [
                    strike,
                    gbm_results[strike][0],
                    gbm_results[strike][1],
                    rho,
                    round(result.price, 8),
                    round(result.standard_error, 8),
                    average_heston_variance,
                    variance_standard_error,
                    variance_ci_half_width,
                ]
            )

    return rows


def build_rho_sensitivity_sheet(rows: list[list[Any]]) -> Sheet:
    return build_correlation_sheet(
        "Heston - change Rho",
        rows,
        RHO_SENSITIVITY_S0,
        RHO_SENSITIVITY_STRIKES,
        MATURITY,
        VOL_OF_VOL_SENSITIVITY_THETA,
        HESTON_KAPPA,
        RHO_SENSITIVITY_VOL_OF_VOL,
        RHO_SENSITIVITY_VALUES,
        RHO_SENSITIVITY_N_PATHS,
    )


def build_correlation_sensitivity_sheet(rows: list[list[Any]]) -> Sheet:
    return build_correlation_sheet(
        "Heston - cng Corr",
        rows,
        CORRELATION_SENSITIVITY_S0,
        CORRELATION_SENSITIVITY_STRIKES,
        MATURITY,
        CORRELATION_SENSITIVITY_THETA,
        CORRELATION_SENSITIVITY_KAPPA,
        CORRELATION_SENSITIVITY_VOL_OF_VOL,
        CORRELATION_SENSITIVITY_VALUES,
        CORRELATION_SENSITIVITY_N_PATHS,
    )


def build_correlation_sheet(
    sheet_name: str,
    rows: list[list[Any]],
    s0: float,
    strikes: tuple[float, ...],
    maturity: float,
    theta: float,
    kappa: float,
    vol_of_vol: float,
    rho_values: tuple[float, ...],
    n_paths: int,
) -> Sheet:
    data_rows = []
    for strike in strikes:
        row = [strike]
        gbm_row = next(item for item in rows if item[0] == strike)
        row.extend([gbm_row[1], gbm_row[2]])
        for rho in rho_values:
            heston_row = next(item for item in rows if item[0] == strike and item[3] == rho)
            row.extend([heston_row[4], heston_row[5]])
        data_rows.append(row)

    headers = ["Strike", "GBM Estimated PV", "GBM Standard Error"]
    for rho in rho_values:
        headers.extend([f"Heston Rho {format_float(rho)} Estimated PV", f"Heston Rho {format_float(rho)} Standard Error"])

    gbm_total_variance = round(GBM_SIGMA * GBM_SIGMA * maturity, 8)
    variance_headers = ["Strike", "GBM"] + [
        f"Rho {format_float(rho)}"
        for rho in rho_values
    ]
    variance_rows = []
    for strike in strikes:
        row = [strike, gbm_total_variance]
        for rho in rho_values:
            heston_row = next(item for item in rows if item[0] == strike and item[3] == rho)
            row.append(heston_row[6])
        variance_rows.append(row)

    variance_chart_headers = [
        "Rho",
        "GBM Variance",
        "Heston Average Variance",
        "Heston Variance Standard Error",
        "Heston Variance 95% CI Half Width",
    ]
    variance_chart_rows = [
        [
            f"Rho {format_float(rho)}",
            gbm_total_variance,
            next(item for item in rows if item[3] == rho)[6],
            next(item for item in rows if item[3] == rho)[7],
            next(item for item in rows if item[3] == rho)[8],
        ]
        for rho in rho_values
    ]

    strike_price_headers = ["Model / Rho", "Estimated PV", "Standard Error", "95% CI Half Width"]
    strike_price_sections = []
    for strike in strikes:
        gbm_row = next(item for item in rows if item[0] == strike)
        section_rows = [
            [
                "GBM",
                gbm_row[1],
                gbm_row[2],
                confidence_interval_half_width(gbm_row[2]),
            ]
        ]
        for rho in rho_values:
            heston_row = next(item for item in rows if item[0] == strike and item[3] == rho)
            section_rows.append(
                [
                    f"Heston rho {format_float(rho)}",
                    heston_row[4],
                    heston_row[5],
                    confidence_interval_half_width(heston_row[5]),
                ]
            )
        strike_price_sections.append((strike, section_rows))

    path_counts = (n_paths,)
    rho_display = ", ".join(format_float(rho) for rho in rho_values)
    data_start = first_data_row(
        path_counts,
        heston_theta=theta,
        heston_vol_of_vol=vol_of_vol,
        heston_kappa=kappa,
        heston_rho=rho_display,
        s0=s0,
        strikes=strikes,
        maturity=maturity,
    )
    data_end = data_start + len(data_rows) - 1
    series = [
        (
            "GBM",
            f"'{sheet_name}'!$A${data_start}:$A${data_end}",
            f"'{sheet_name}'!$B${data_start}:$B${data_end}",
            f"'{sheet_name}'!$C${data_start}:$C${data_end}",
        )
    ]
    for index, rho in enumerate(rho_values):
        value_column = _column_name(4 + index * 2)
        error_column = _column_name(5 + index * 2)
        series.append(
            (
                f"Heston rho {format_float(rho)}",
                f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                f"'{sheet_name}'!${value_column}${data_start}:${value_column}${data_end}",
                f"'{sheet_name}'!${error_column}${data_start}:${error_column}${data_end}",
            )
        )

    price_chart: ChartSpec = (
        f"GBM and Heston prices by rho, {n_paths:,} paths",
        1,
        18,
        series,
    )
    worksheet_rows = (
        parameter_rows(
            path_counts,
            heston_theta=theta,
            heston_vol_of_vol=vol_of_vol,
            heston_kappa=kappa,
            heston_rho=rho_display,
            s0=s0,
            strikes=strikes,
            maturity=maturity,
        )
        + [[]]
        + [headers]
        + data_rows
        + [[]]
        + [variance_headers]
        + variance_rows
        + [[]]
        + [variance_chart_headers]
        + variance_chart_rows
    )
    strike_price_section_ranges = []
    for strike, strike_price_rows in strike_price_sections:
        worksheet_rows.append([])
        worksheet_rows.append([f"Price by model and rho for strike {strike:.2f}"])
        worksheet_rows.append(strike_price_headers)
        section_data_start = len(worksheet_rows) + 1
        worksheet_rows.extend(strike_price_rows)
        section_data_end = section_data_start + len(strike_price_rows) - 1
        strike_price_section_ranges.append((strike, section_data_start, section_data_end))

    variance_chart_header_row = data_end + 2 + 1 + len(variance_rows) + 1
    variance_chart_data_start = variance_chart_header_row + 1
    variance_chart_data_end = variance_chart_data_start + len(variance_chart_rows) - 1
    variance_chart: ChartSpec = (
        f"GBM and Heston average variance by rho, {n_paths:,} paths",
        17,
        18,
        [
            (
                "GBM variance",
                f"'{sheet_name}'!$A${variance_chart_data_start}:$A${variance_chart_data_end}",
                f"'{sheet_name}'!$B${variance_chart_data_start}:$B${variance_chart_data_end}",
                "",
            ),
            (
                "Heston average variance",
                f"'{sheet_name}'!$A${variance_chart_data_start}:$A${variance_chart_data_end}",
                f"'{sheet_name}'!$C${variance_chart_data_start}:$C${variance_chart_data_end}",
                f"'{sheet_name}'!$E${variance_chart_data_start}:$E${variance_chart_data_end}",
            ),
        ],
    )
    strike_price_charts: list[ChartSpec] = []
    for strike_index, (strike, section_data_start, section_data_end) in enumerate(strike_price_section_ranges):
        strike_price_charts.append(
            (
                f"Price by GBM and Heston rho, strike={strike:.2f}",
                33 + strike_index * 16,
                18,
                [
                    (
                        f"Strike {strike:.2f}",
                        f"'{sheet_name}'!$A${section_data_start}:$A${section_data_end}",
                        f"'{sheet_name}'!$B${section_data_start}:$B${section_data_end}",
                        f"'{sheet_name}'!$D${section_data_start}:$D${section_data_end}",
                    )
                ],
            )
        )

    return (sheet_name, [], worksheet_rows, [price_chart, variance_chart] + strike_price_charts)


def _column_name(column_index: int) -> str:
    name = ""
    while column_index:
        column_index, remainder = divmod(column_index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def build_workbook_sheets() -> list[Sheet]:
    print("Calculating GBM results...")
    gbm_rows = calculate_model_results("GBM")
    print("Calculating Heston results...")
    heston_rows, average_variance_rows, path_variance_rows = calculate_heston_results()
    print("Calculating theta sensitivity results...")
    theta_sensitivity_rows = calculate_theta_sensitivity_rows()
    print("Calculating vol_of_vol sensitivity results...")
    vol_of_vol_sensitivity_rows = calculate_vol_of_vol_sensitivity_rows()
    print(f"Calculating vol_of_vol sensitivity results for rho={NEGATIVE_CORRELATION_VOL_OF_VOL:g}...")
    negative_correlation_vol_of_vol_rows = calculate_negative_correlation_vol_of_vol_sensitivity_rows()
    print("Calculating kappa sensitivity results for rho=0.3...")
    kappa_sensitivity_rows_corr_0_3 = calculate_kappa_sensitivity_rows_corr_0_3()
    print(f"Calculating kappa sensitivity results for rho={NEGATIVE_CORRELATION_VOL_OF_VOL:g}...")
    kappa_sensitivity_rows_corr_negative_0_7 = calculate_kappa_sensitivity_rows_corr_negative_0_7()
    print("Calculating correlation sensitivity results...")
    correlation_sensitivity_rows = calculate_correlation_sensitivity_rows()

    largest_n_paths = max(N_PATH_VALUES)
    return [
        build_summary_sheet(f"Base GBM vs Heston {largest_n_paths:,}", gbm_rows, heston_rows),
        build_convergence_sheet("Base GBM Convergence", gbm_rows),
        build_convergence_sheet("Base Heston Convergence", heston_rows),
        build_average_variance_sheet(average_variance_rows),
        build_path_variance_sheet(path_variance_rows),
        build_theta_sensitivity_sheet(theta_sensitivity_rows),
        build_vol_of_vol_sensitivity_sheet(vol_of_vol_sensitivity_rows),
        build_negative_correlation_vol_of_vol_sensitivity_sheet(negative_correlation_vol_of_vol_rows),
        build_kappa_sensitivity_sheet_corr_0_3(kappa_sensitivity_rows_corr_0_3),
        build_kappa_sensitivity_sheet_corr_negative_0_7(kappa_sensitivity_rows_corr_negative_0_7),
        build_correlation_sensitivity_sheet(correlation_sensitivity_rows),
    ]


def main() -> None:
    sheets = build_workbook_sheets()
    write_workbook(DEFAULT_OUTPUT_PATH, sheets)
    print(f"Excel file saved to {DEFAULT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
