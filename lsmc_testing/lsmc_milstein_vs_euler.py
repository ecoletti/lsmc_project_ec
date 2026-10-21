"""Compare Heston LSMC convergence under Euler-Maruyama and Milstein."""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any

from lsmc_testing.black_scholes import american_put_price_barone_adesi_whaley, european_put_price
from lsmc_testing.excel import ChartSpec, Sheet, write_workbook
from lsmc_testing.pricing import price_american_put_lsm
from lsmc_testing.simulation import simulate_gbm_paths, simulate_heston_spot_scaled_vol_of_vol_paths


MATURITY = 1.0
RATES = (0.0, 0.06)
DIVIDEND = 0.0
NUMBER_OF_TIME_STEPS = 250
TIME_STEP_COUNTS = (50, 100, 150, 200, 250)
LAGUERRE_DEGREE = 2
LAGUERRE_DEGREES = (2, 3, 4)
PATH_COUNTS = (10_000, 20_000, 40_000, 100_000)
DT_CONVERGENCE_PATH_COUNT = 100_000
DEGREE_CONVERGENCE_PATH_COUNT = 100_000
STRIKE = 4.0
SPOT_LEVELS = (3.6, 3.8, 4.0, 4.2, 4.4)
UNIT_STRIKE = 1.0
UNIT_SPOT_LEVELS = (0.9, 0.95, 1.0, 1.05, 1.1)
UNIT_DT_SPOT = 1.1
UNIT_DT_STRIKES = (1.1,)
UNIT_DT_CORRELATION = -0.3
UNIT_DT_HESTON_V0 = 0.04
UNIT_DT_HESTON_THETA = 0.04
UNIT_DT_HESTON_VOL_OF_VOL = 0.35
UNIT_MC_SPOT = 1.0
UNIT_MC_STRIKES = (0.5, 0.75, 1.0, 1.25, 1.5)
UNIT_MC_CORRELATION = 0.3
UNIT_MC_HESTON_V0 = 0.08
UNIT_MC_HESTON_THETA = 0.08
UNIT_MC_HESTON_VOL_OF_VOL = 0.4
UNIT_MC_TIME_STEPS = 50
UNIT_MC_PATH_COUNTS = (1_000, 5_000, 10_000, 20_000, 40_000, 100_000)
UNIT_MC_RATE = 0.06
UNIT_REFERENCE_PATH_COUNT = 500_000
UNIT_REFERENCE_TIME_STEPS = 500
UNIT_TIME_CONVERGENCE_PATH_COUNT = 100_000
UNIT_TIME_STEP_COUNTS = (12, 25, 50, 100, 150, 200, 250, 300, 350, 400)
UNIT_MC_CONVERGENCE_TIME_STEPS = 250
UNIT_MC_CONVERGENCE_PATH_COUNTS = (1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000)
HESTON_V0 = 0.01
HESTON_THETA = 0.01
HESTON_CORRELATIONS = (0.3, -0.3)
HESTON_KAPPA = 2.0
HESTON_VOL_OF_VOL = 0.15
BASE_SEED = 20260929
DEFAULT_OUTPUT_PATH = Path("output") / "lsmc_milstein_vs_euler.xlsx"


def laguerre_polynomial_description(degree: int) -> str:
    polynomials = [
        "e^(-x/2)",
        "e^(-x/2)*(1-x)",
        "e^(-x/2)*(1-2x+x^(2)/2)",
        "e^(-x/2)*(1-3x+3x^(2)/2-x^(3)/6)",
    ]
    return "[" + ", ".join(polynomials[: degree + 1]) + "]"


def sheet_number(value: float) -> str:
    text = f"{value:g}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def full_number(value: float) -> str:
    return f"{value:g}"


def rate_sheet_name(rate: float, correlation: float) -> str:
    return f"Base Mlst_Eul r={full_number(rate)} corr={full_number(correlation)}"


def dt_convergence_sheet_name(rate: float, correlation: float) -> str:
    full_name = f"Mlst_Eul r={full_number(rate)} corr={full_number(correlation)} Dt conv"
    if len(full_name) <= 31:
        return full_name
    return f"Mlst_Eul r={sheet_number(rate)} corr={sheet_number(correlation)} Dt conv"


def average_variance_sheet_name(correlation: float) -> str:
    return f"Heston avg var corr={full_number(correlation)}"


def parameter_rows(
    rate: float,
    correlation: float,
    *,
    strike: float = STRIKE,
    spot_levels: tuple[float, ...] = SPOT_LEVELS,
) -> list[list[Any]]:
    return [
        ["Strike", strike],
        ["Maturity", MATURITY],
        ["Rate", rate],
        ["Dividend", DIVIDEND],
        ["Number of time steps", NUMBER_OF_TIME_STEPS],
        ["Polynomials", laguerre_polynomial_description(LAGUERRE_DEGREE)],
        ["Number of MC paths", ", ".join(f"{path_count:,}" for path_count in PATH_COUNTS)],
        ["Spot initial value", ", ".join(f"{spot:g}" for spot in spot_levels)],
        ["Heston v0", HESTON_V0],
        ["Heston theta", HESTON_THETA],
        ["Heston rho", correlation],
        ["Heston kappa", HESTON_KAPPA],
        ["Heston vol of vol", HESTON_VOL_OF_VOL],
        ["Reference volatility", "sqrt(Heston theta)"],
        [],
    ]


def dt_convergence_parameter_rows(
    rate: float,
    correlation: float,
    *,
    strike: float = STRIKE,
    spot_levels: tuple[float, ...] = SPOT_LEVELS,
) -> list[list[Any]]:
    return [
        ["Strike", strike],
        ["Maturity", MATURITY],
        ["Rate", rate],
        ["Dividend", DIVIDEND],
        ["Number of time steps", ", ".join(str(step_count) for step_count in TIME_STEP_COUNTS)],
        ["Polynomials", laguerre_polynomial_description(LAGUERRE_DEGREE)],
        ["Number of MC paths", f"{DT_CONVERGENCE_PATH_COUNT:,}"],
        ["Spot initial value", ", ".join(f"{spot:g}" for spot in spot_levels)],
        ["Heston v0", HESTON_V0],
        ["Heston theta", HESTON_THETA],
        ["Heston rho", correlation],
        ["Heston kappa", HESTON_KAPPA],
        ["Heston vol of vol", HESTON_VOL_OF_VOL],
        ["Reference volatility", "sqrt(Heston theta)"],
        [],
    ]


def calculate_heston_lsm_price(
    *,
    s0: float,
    rate: float,
    n_paths: int,
    seed: int,
    variance_scheme: str,
    correlation: float,
    time_steps: int = NUMBER_OF_TIME_STEPS,
    laguerre_degree: int = LAGUERRE_DEGREE,
    strike: float = STRIKE,
    heston_v0: float = HESTON_V0,
    heston_theta: float = HESTON_THETA,
    heston_vol_of_vol: float = HESTON_VOL_OF_VOL,
) -> tuple[float, float]:
    paths = simulate_heston_spot_scaled_vol_of_vol_paths(
        s0=s0,
        rate=rate,
        v0=heston_v0,
        kappa=HESTON_KAPPA,
        theta=heston_theta,
        vol_of_vol=heston_vol_of_vol,
        rho=correlation,
        maturity=MATURITY,
        steps_per_year=time_steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=True,
        variance_scheme=variance_scheme,
    )
    result = price_american_put_lsm(
        paths,
        strike=strike,
        rate=rate,
        maturity=MATURITY,
        laguerre_degree=laguerre_degree,
    )
    return round(result.price, 8), round(result.standard_error, 8)


def calculate_gbm_lsm_price(
    *,
    s0: float,
    rate: float,
    n_paths: int,
    seed: int,
    time_steps: int = NUMBER_OF_TIME_STEPS,
    laguerre_degree: int = LAGUERRE_DEGREE,
    strike: float = STRIKE,
    reference_theta: float = HESTON_THETA,
) -> tuple[float, float]:
    paths = simulate_gbm_paths(
        s0=s0,
        rate=rate,
        sigma=sqrt(reference_theta),
        maturity=MATURITY,
        steps_per_year=time_steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=True,
    )
    result = price_american_put_lsm(
        paths,
        strike=strike,
        rate=rate,
        maturity=MATURITY,
        laguerre_degree=laguerre_degree,
    )
    return round(result.price, 8), round(result.standard_error, 8)


def total_variance_by_path(variance_paths: Any) -> Any:
    return variance_paths[1:].mean(axis=0) * MATURITY


def calculate_average_heston_variance(
    *,
    s0: float,
    rate: float,
    n_paths: int,
    seed: int,
    variance_scheme: str,
    correlation: float,
) -> tuple[float, float, float]:
    _, variance_paths, floor_hit_rate = simulate_heston_spot_scaled_vol_of_vol_paths(
        s0=s0,
        rate=rate,
        v0=HESTON_V0,
        kappa=HESTON_KAPPA,
        theta=HESTON_THETA,
        vol_of_vol=HESTON_VOL_OF_VOL,
        rho=correlation,
        maturity=MATURITY,
        steps_per_year=NUMBER_OF_TIME_STEPS,
        n_paths=n_paths,
        seed=seed,
        antithetic=True,
        variance_scheme=variance_scheme,
        return_variance=True,
        return_floor_hit_rate=True,
    )
    total_variances = total_variance_by_path(variance_paths)
    standard_error = float(total_variances.std(ddof=1) / sqrt(n_paths))
    return round(float(total_variances.mean()), 8), round(standard_error, 8), round(floor_hit_rate, 8)


def calculate_rows_for_spot(
    s0: float,
    rate: float,
    correlation: float,
    spot_index: int,
    strike: float = STRIKE,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    reference_sigma = sqrt(HESTON_THETA)
    baw_price = round(
        american_put_price_barone_adesi_whaley(
            s0,
            strike,
            rate,
            reference_sigma,
            MATURITY,
            dividend_yield=DIVIDEND,
        ),
        8,
    )
    european_price = round(european_put_price(s0, strike, rate, reference_sigma, MATURITY), 8)

    for path_index, n_paths in enumerate(PATH_COUNTS):
        seed = (
            BASE_SEED
            + int(rate * 1_000_000)
            + int((correlation + 1.0) * 1_000_000)
            + spot_index * 10_000
            + path_index
        )
        print(f"Calculating r={rate:g}, corr={correlation:g}, S0={s0:g}, paths={n_paths:,}")
        euler_price, euler_se = calculate_heston_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=n_paths,
            seed=seed,
            variance_scheme="euler",
            correlation=correlation,
            strike=strike,
        )
        milstein_price, milstein_se = calculate_heston_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=n_paths,
            seed=seed,
            variance_scheme="milstein",
            correlation=correlation,
            strike=strike,
        )
        gbm_price, gbm_se = calculate_gbm_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=n_paths,
            seed=seed,
            strike=strike,
        )
        rows.append(
            [
                n_paths,
                euler_price,
                euler_se,
                milstein_price,
                milstein_se,
                gbm_price,
                gbm_se,
                baw_price,
                european_price,
            ]
        )
    return rows


def calculate_dt_convergence_rows_for_spot(
    s0: float,
    rate: float,
    correlation: float,
    spot_index: int,
    strike: float = STRIKE,
    heston_v0: float = HESTON_V0,
    heston_theta: float = HESTON_THETA,
    heston_vol_of_vol: float = HESTON_VOL_OF_VOL,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    reference_sigma = sqrt(heston_theta)
    baw_price = round(
        american_put_price_barone_adesi_whaley(
            s0,
            strike,
            rate,
            reference_sigma,
            MATURITY,
            dividend_yield=DIVIDEND,
        ),
        8,
    )
    european_price = round(european_put_price(s0, strike, rate, reference_sigma, MATURITY), 8)

    for time_step_index, time_steps in enumerate(TIME_STEP_COUNTS):
        seed = (
            BASE_SEED
            + 7_000_000
            + int(rate * 1_000_000)
            + int((correlation + 1.0) * 1_000_000)
            + spot_index * 10_000
            + time_step_index
        )
        print(
            f"Calculating Dt convergence r={rate:g}, corr={correlation:g}, S0={s0:g}, "
            f"steps={time_steps:,}, paths={DT_CONVERGENCE_PATH_COUNT:,}"
        )
        euler_price, euler_se = calculate_heston_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=DT_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="euler",
            correlation=correlation,
            time_steps=time_steps,
            strike=strike,
            heston_v0=heston_v0,
            heston_theta=heston_theta,
            heston_vol_of_vol=heston_vol_of_vol,
        )
        milstein_price, milstein_se = calculate_heston_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=DT_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="milstein",
            correlation=correlation,
            time_steps=time_steps,
            strike=strike,
            heston_v0=heston_v0,
            heston_theta=heston_theta,
            heston_vol_of_vol=heston_vol_of_vol,
        )
        gbm_price, gbm_se = calculate_gbm_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=DT_CONVERGENCE_PATH_COUNT,
            seed=seed,
            time_steps=time_steps,
            strike=strike,
            reference_theta=heston_theta,
        )
        rows.append(
            [
                time_steps,
                euler_price,
                euler_se,
                milstein_price,
                milstein_se,
                gbm_price,
                gbm_se,
                baw_price,
                european_price,
            ]
        )
    return rows


def calculate_dt_convergence_rows_for_strike(
    *,
    strike: float,
    strike_index: int,
    s0: float,
    rate: float,
    correlation: float,
    heston_v0: float,
    heston_theta: float,
    heston_vol_of_vol: float,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    reference_sigma = sqrt(heston_theta)
    baw_price = round(
        american_put_price_barone_adesi_whaley(
            s0,
            strike,
            rate,
            reference_sigma,
            MATURITY,
            dividend_yield=DIVIDEND,
        ),
        8,
    )
    european_price = round(european_put_price(s0, strike, rate, reference_sigma, MATURITY), 8)

    for time_step_index, time_steps in enumerate(TIME_STEP_COUNTS):
        seed = (
            BASE_SEED
            + 13_000_000
            + int(rate * 1_000_000)
            + int((correlation + 1.0) * 1_000_000)
            + strike_index * 10_000
            + time_step_index
        )
        print(
            f"Calculating unit strike Dt convergence r={rate:g}, corr={correlation:g}, "
            f"S0={s0:g}, K={strike:g}, steps={time_steps:,}, paths={DT_CONVERGENCE_PATH_COUNT:,}"
        )
        euler_price, euler_se = calculate_heston_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=DT_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="euler",
            correlation=correlation,
            time_steps=time_steps,
            strike=strike,
            heston_v0=heston_v0,
            heston_theta=heston_theta,
            heston_vol_of_vol=heston_vol_of_vol,
        )
        milstein_price, milstein_se = calculate_heston_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=DT_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="milstein",
            correlation=correlation,
            time_steps=time_steps,
            strike=strike,
            heston_v0=heston_v0,
            heston_theta=heston_theta,
            heston_vol_of_vol=heston_vol_of_vol,
        )
        gbm_price, gbm_se = calculate_gbm_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=DT_CONVERGENCE_PATH_COUNT,
            seed=seed,
            time_steps=time_steps,
            strike=strike,
            reference_theta=heston_theta,
        )
        rows.append(
            [
                time_steps,
                euler_price,
                euler_se,
                milstein_price,
                milstein_se,
                gbm_price,
                gbm_se,
                baw_price,
                european_price,
            ]
        )
    return rows


def padded_bounds(values: list[float], padding_fraction: float = 0.15) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    padding = max((max_value - min_value) * padding_fraction, abs(max_value) * 0.05, 0.01)
    lower = min_value - padding
    upper = max_value + padding
    if lower == upper:
        upper = lower + 1.0
    return round(lower, 8), round(upper, 8)


def exact_bounds(values: list[float]) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if lower == upper:
        padding = max(abs(upper) * 0.01, 0.00000001)
        lower -= padding
        upper += padding
    return round(lower, 8), round(upper, 8)


def build_rate_sheet(
    rate: float,
    correlation: float,
    *,
    sheet_name: str | None = None,
    sheet_title: str | None = None,
    strike: float = STRIKE,
    spot_levels: tuple[float, ...] = SPOT_LEVELS,
) -> Sheet:
    sheet_name = sheet_name or rate_sheet_name(rate, correlation)
    headers = [
        "Number of MC paths",
        "Heston American Put Price using Euler-Maruyama",
        "Euler-Maruyama s.e.",
        "Heston American Put Price using Milstein",
        "Milstein s.e.",
        "GBM American Put Price",
        "GBM s.e.",
        "BAW Approx Price",
        "European Put Price",
    ]
    rows = [[sheet_title or f"Base Mlistein vs Euler r={full_number(rate)} corr={full_number(correlation)}"], []]
    rows.extend(parameter_rows(rate, correlation, strike=strike, spot_levels=spot_levels))
    charts: list[ChartSpec] = []
    spot_profile_rows: list[list[Any]] = [
        [
            "Spot",
            "Euler-Maruyama",
            "Euler-Maruyama s.e.",
            "Milstein",
            "Milstein s.e.",
            "GBM",
            "GBM s.e.",
            "BAW Approx Price",
            "European Put Price",
        ]
    ]

    for spot_index, s0 in enumerate(spot_levels):
        block_start = len(rows) + 1
        table_rows = calculate_rows_for_spot(s0, rate, correlation, spot_index, strike=strike)
        final_path_row = table_rows[-1]
        spot_profile_rows.append(
            [
                s0,
                final_path_row[1],
                final_path_row[2],
                final_path_row[3],
                final_path_row[4],
                final_path_row[5],
                final_path_row[6],
                final_path_row[7],
                final_path_row[8],
            ]
        )
        rows.append([f"Initial spot {s0:g}"])
        rows.append(headers)
        rows.extend(table_rows)
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + len(PATH_COUNTS) - 1
        price_bounds = [
            value
            for row in table_rows
            for value in (
                row[1] - row[2],
                row[1] + row[2],
                row[3] - row[4],
                row[3] + row[4],
                row[5] - row[6],
                row[5] + row[6],
                row[7],
                row[8],
            )
        ]
        y_axis_min, y_axis_max = padded_bounds([float(value) for value in price_bounds])
        charts.append(
            (
                f"Milstein vs Euler convergence, S0={s0:g}",
                1 + spot_index * 18,
                10,
                [
                    (
                        "Euler-Maruyama",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    ),
                    (
                        "Milstein",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                    ),
                    (
                        "GBM",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                        f"'{sheet_name}'!$G${data_start}:$G${data_end}",
                    ),
                    (
                        "BAW approximation",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$H${data_start}:$H${data_end}",
                        "",
                    ),
                    (
                        "European put",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$I${data_start}:$I${data_end}",
                        "",
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Number of MC paths",
                "American put price",
                "0",
                "0.000000",
            )
        )

    profile_start = len(rows) + 1
    rows.append(["100,000 path price profile by spot"])
    rows.extend(spot_profile_rows)
    profile_header_row = profile_start + 1
    profile_data_start = profile_header_row + 1
    profile_data_end = profile_data_start + len(spot_levels) - 1
    profile_bounds = [
        value
        for row in spot_profile_rows[1:]
        for value in (
            row[1] - row[2],
            row[1] + row[2],
            row[3] - row[4],
            row[3] + row[4],
            row[5] - row[6],
            row[5] + row[6],
            row[7],
            row[8],
        )
    ]
    profile_y_axis_min, profile_y_axis_max = padded_bounds([float(value) for value in profile_bounds])
    charts.append(
        (
            "100,000 path prices by spot",
            1,
            21,
            [
                (
                    "Heston Euler-Maruyama",
                    f"'{sheet_name}'!$A${profile_data_start}:$A${profile_data_end}",
                    f"'{sheet_name}'!$B${profile_data_start}:$B${profile_data_end}",
                    f"'{sheet_name}'!$C${profile_data_start}:$C${profile_data_end}",
                ),
                (
                    "Heston Milstein",
                    f"'{sheet_name}'!$A${profile_data_start}:$A${profile_data_end}",
                    f"'{sheet_name}'!$D${profile_data_start}:$D${profile_data_end}",
                    f"'{sheet_name}'!$E${profile_data_start}:$E${profile_data_end}",
                ),
                (
                    "GBM",
                    f"'{sheet_name}'!$A${profile_data_start}:$A${profile_data_end}",
                    f"'{sheet_name}'!$F${profile_data_start}:$F${profile_data_end}",
                    f"'{sheet_name}'!$G${profile_data_start}:$G${profile_data_end}",
                ),
                (
                    "BAW approximation",
                    f"'{sheet_name}'!$A${profile_data_start}:$A${profile_data_end}",
                    f"'{sheet_name}'!$H${profile_data_start}:$H${profile_data_end}",
                    "",
                ),
                (
                    "European put",
                    f"'{sheet_name}'!$A${profile_data_start}:$A${profile_data_end}",
                    f"'{sheet_name}'!$I${profile_data_start}:$I${profile_data_end}",
                    "",
                ),
            ],
            "line",
            profile_y_axis_min,
            profile_y_axis_max,
            "Spot",
            "American put price",
            "0",
            "0.000000",
        )
    )

    return (sheet_name, [], rows, charts)


def build_dt_convergence_sheet(
    rate: float,
    correlation: float,
    *,
    sheet_name: str | None = None,
    sheet_title: str | None = None,
    strike: float = STRIKE,
    spot_levels: tuple[float, ...] = SPOT_LEVELS,
) -> Sheet:
    sheet_name = sheet_name or dt_convergence_sheet_name(rate, correlation)
    sheet_title = sheet_title or (
        f"Base Mlistein vs Euler r={full_number(rate)} "
        f"corr={full_number(correlation)} Dt conv"
    )
    headers = [
        "Number of time discretization steps",
        "Heston American Put Price using Euler-Maruyama",
        "Euler-Maruyama s.e.",
        "Heston American Put Price using Milstein",
        "Milstein s.e.",
        "GBM American Put Price",
        "GBM s.e.",
        "BAW Approx Price",
        "European Put Price",
    ]
    rows = [[sheet_title], []]
    rows.extend(dt_convergence_parameter_rows(rate, correlation, strike=strike, spot_levels=spot_levels))
    charts: list[ChartSpec] = []

    for spot_index, s0 in enumerate(spot_levels):
        block_start = len(rows) + 1
        table_rows = calculate_dt_convergence_rows_for_spot(
            s0,
            rate,
            correlation,
            spot_index,
            strike=strike,
        )
        rows.append([f"Initial spot {s0:g}"])
        rows.append(headers)
        rows.extend(table_rows)
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + len(TIME_STEP_COUNTS) - 1
        price_bounds = [
            value
            for row in table_rows
            for value in (
                row[1] - row[2],
                row[1] + row[2],
                row[3] - row[4],
                row[3] + row[4],
                row[5] - row[6],
                row[5] + row[6],
                row[7],
                row[8],
            )
        ]
        y_axis_min, y_axis_max = padded_bounds([float(value) for value in price_bounds])
        charts.append(
            (
                f"Milstein vs Euler Dt convergence, S0={s0:g}",
                1 + spot_index * 18,
                10,
                [
                    (
                        "Euler-Maruyama",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    ),
                    (
                        "Milstein",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                    ),
                    (
                        "GBM",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                        f"'{sheet_name}'!$G${data_start}:$G${data_end}",
                    ),
                    (
                        "BAW approximation",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$H${data_start}:$H${data_end}",
                        "",
                    ),
                    (
                        "European put",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$I${data_start}:$I${data_end}",
                        "",
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Number of time discretization steps",
                "American put price",
                "0",
                "0.000000",
            )
        )

    return (sheet_name, [], rows, charts)


def build_unit_strike_dt_convergence_sheet() -> Sheet:
    sheet_name = "unit S_K Mlst_Eul Dt conv"
    headers = [
        "Number of time discretization steps",
        "Heston American Put Price using Euler-Maruyama",
        "Euler-Maruyama s.e.",
        "Heston American Put Price using Milstein",
        "Milstein s.e.",
        "GBM American Put Price",
        "GBM s.e.",
        "BAW Approx Price",
        "European Put Price",
    ]
    rows: list[list[Any]] = [
        ["unit Spot_strike Mlst_Eul r=0 corr=-0.3 Dt conv"],
        [],
        ["Strike", ", ".join(f"{strike:g}" for strike in UNIT_DT_STRIKES)],
        ["Spot initial value", UNIT_DT_SPOT],
        ["Maturity", MATURITY],
        ["Rate", UNIT_MC_RATE],
        ["Dividend", DIVIDEND],
        ["Number of time steps", ", ".join(str(step_count) for step_count in TIME_STEP_COUNTS)],
        ["Polynomials", laguerre_polynomial_description(LAGUERRE_DEGREE)],
        ["Number of MC paths", f"{DT_CONVERGENCE_PATH_COUNT:,}"],
        ["Heston v0", UNIT_DT_HESTON_V0],
        ["Heston theta", UNIT_DT_HESTON_THETA],
        ["Heston rho", UNIT_DT_CORRELATION],
        ["Heston kappa", HESTON_KAPPA],
        ["Heston vol of vol", UNIT_DT_HESTON_VOL_OF_VOL],
        ["Reference volatility", "sqrt(Heston theta)"],
        [],
    ]
    charts: list[ChartSpec] = []

    for strike_index, strike in enumerate(UNIT_DT_STRIKES):
        block_start = len(rows) + 1
        table_rows = calculate_dt_convergence_rows_for_strike(
            strike=strike,
            strike_index=strike_index,
            s0=UNIT_DT_SPOT,
            rate=0.0,
            correlation=UNIT_DT_CORRELATION,
            heston_v0=UNIT_DT_HESTON_V0,
            heston_theta=UNIT_DT_HESTON_THETA,
            heston_vol_of_vol=UNIT_DT_HESTON_VOL_OF_VOL,
        )
        rows.append([f"Strike {strike:g}"])
        rows.append(headers)
        rows.extend(table_rows)
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + len(TIME_STEP_COUNTS) - 1
        price_bounds = [
            value
            for row in table_rows
            for value in (
                row[1] - row[2],
                row[1] + row[2],
                row[3] - row[4],
                row[3] + row[4],
                row[5] - row[6],
                row[5] + row[6],
                row[7],
                row[8],
            )
        ]
        y_axis_min, y_axis_max = padded_bounds([float(value) for value in price_bounds])
        charts.append(
            (
                f"Milstein vs Euler Dt convergence, K={strike:g}",
                1 + strike_index * 18,
                10,
                [
                    (
                        "Euler-Maruyama",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    ),
                    (
                        "Milstein",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                    ),
                    (
                        "GBM",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                        f"'{sheet_name}'!$G${data_start}:$G${data_end}",
                    ),
                    (
                        "BAW approximation",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$H${data_start}:$H${data_end}",
                        "",
                    ),
                    (
                        "European put",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$I${data_start}:$I${data_end}",
                        "",
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Number of time discretization steps",
                "American put price",
                "0",
                "0.000000",
            )
        )

    return (sheet_name, [], rows, charts)


def calculate_mc_convergence_rows_for_strike(
    *,
    strike: float,
    strike_index: int,
    s0: float,
    rate: float,
    correlation: float,
    heston_v0: float,
    heston_theta: float,
    heston_vol_of_vol: float,
    time_steps: int,
    path_counts: tuple[int, ...],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    reference_sigma = sqrt(heston_theta)
    baw_price = round(
        american_put_price_barone_adesi_whaley(
            s0,
            strike,
            rate,
            reference_sigma,
            MATURITY,
            dividend_yield=DIVIDEND,
        ),
        8,
    )
    european_price = round(european_put_price(s0, strike, rate, reference_sigma, MATURITY), 8)

    for path_index, n_paths in enumerate(path_counts):
        seed = (
            BASE_SEED
            + 14_000_000
            + int(rate * 1_000_000)
            + int((correlation + 1.0) * 1_000_000)
            + strike_index * 10_000
            + path_index
        )
        print(
            f"Calculating unit strike MC convergence r={rate:g}, corr={correlation:g}, "
            f"S0={s0:g}, K={strike:g}, paths={n_paths:,}"
        )
        euler_price, euler_se = calculate_heston_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=n_paths,
            seed=seed,
            variance_scheme="euler",
            correlation=correlation,
            time_steps=time_steps,
            strike=strike,
            heston_v0=heston_v0,
            heston_theta=heston_theta,
            heston_vol_of_vol=heston_vol_of_vol,
        )
        milstein_price, milstein_se = calculate_heston_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=n_paths,
            seed=seed,
            variance_scheme="milstein",
            correlation=correlation,
            time_steps=time_steps,
            strike=strike,
            heston_v0=heston_v0,
            heston_theta=heston_theta,
            heston_vol_of_vol=heston_vol_of_vol,
        )
        gbm_price, gbm_se = calculate_gbm_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=n_paths,
            seed=seed,
            time_steps=time_steps,
            strike=strike,
            reference_theta=heston_theta,
        )
        rows.append(
            [
                n_paths,
                euler_price,
                euler_se,
                milstein_price,
                milstein_se,
                gbm_price,
                gbm_se,
                baw_price,
                european_price,
            ]
        )
    return rows


def build_unit_strike_mc_convergence_sheet(
    *,
    sheet_name: str = "Mlst vs Eul corr=0.3",
    sheet_title: str = "Mlst vs Eul corr=0.3",
    correlation: float = UNIT_MC_CORRELATION,
    reference_rows: list[list[Any]] | None = None,
) -> Sheet:
    return build_unit_strike_time_convergence_sheet(
        sheet_name=sheet_name,
        sheet_title=sheet_title,
        correlation=correlation,
        reference_rows=reference_rows,
    )


def calculate_reference_rows_for_strike(
    *,
    strike: float,
    strike_index: int,
    correlation: float,
) -> list[Any]:
    seed = (
        BASE_SEED
        + 15_000_000
        + int(UNIT_MC_RATE * 1_000_000)
        + int((correlation + 1.0) * 1_000_000)
        + strike_index * 10_000
    )
    print(
        f"Calculating reference r={UNIT_MC_RATE:g}, corr={correlation:g}, "
        f"S0={UNIT_MC_SPOT:g}, K={strike:g}, steps={UNIT_REFERENCE_TIME_STEPS:,}, "
        f"paths={UNIT_REFERENCE_PATH_COUNT:,}"
    )
    euler_price, euler_se = calculate_heston_lsm_price(
        s0=UNIT_MC_SPOT,
        rate=UNIT_MC_RATE,
        n_paths=UNIT_REFERENCE_PATH_COUNT,
        seed=seed,
        variance_scheme="euler",
        correlation=correlation,
        time_steps=UNIT_REFERENCE_TIME_STEPS,
        strike=strike,
        heston_v0=UNIT_MC_HESTON_V0,
        heston_theta=UNIT_MC_HESTON_THETA,
        heston_vol_of_vol=UNIT_MC_HESTON_VOL_OF_VOL,
    )
    milstein_price, milstein_se = calculate_heston_lsm_price(
        s0=UNIT_MC_SPOT,
        rate=UNIT_MC_RATE,
        n_paths=UNIT_REFERENCE_PATH_COUNT,
        seed=seed,
        variance_scheme="milstein",
        correlation=correlation,
        time_steps=UNIT_REFERENCE_TIME_STEPS,
        strike=strike,
        heston_v0=UNIT_MC_HESTON_V0,
        heston_theta=UNIT_MC_HESTON_THETA,
        heston_vol_of_vol=UNIT_MC_HESTON_VOL_OF_VOL,
    )
    difference = round(milstein_price - euler_price, 8)
    combined_standard_error = round(sqrt(euler_se * euler_se + milstein_se * milstein_se), 8)
    two_standard_error_band = round(2.0 * combined_standard_error, 8)
    return [
        strike,
        euler_price,
        euler_se,
        milstein_price,
        milstein_se,
        difference,
        combined_standard_error,
        two_standard_error_band,
        "Yes" if abs(difference) <= two_standard_error_band else "No",
    ]


def calculate_reference_rows(correlation: float) -> list[list[Any]]:
    return [
        calculate_reference_rows_for_strike(
            strike=strike,
            strike_index=strike_index,
            correlation=correlation,
        )
        for strike_index, strike in enumerate(UNIT_MC_STRIKES)
    ]


def calculate_time_convergence_rows_for_strike(
    *,
    strike: float,
    strike_index: int,
    correlation: float,
    reference_row: list[Any],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for time_steps in UNIT_TIME_STEP_COUNTS:
        seed = (
            BASE_SEED
            + 16_000_000
            + int(UNIT_MC_RATE * 1_000_000)
            + int((correlation + 1.0) * 1_000_000)
            + strike_index * 10_000
        )
        print(
            f"Calculating time-step convergence r={UNIT_MC_RATE:g}, corr={correlation:g}, "
            f"S0={UNIT_MC_SPOT:g}, K={strike:g}, steps={time_steps:,}, "
            f"paths={UNIT_TIME_CONVERGENCE_PATH_COUNT:,}"
        )
        euler_price, euler_se = calculate_heston_lsm_price(
            s0=UNIT_MC_SPOT,
            rate=UNIT_MC_RATE,
            n_paths=UNIT_TIME_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="euler",
            correlation=correlation,
            time_steps=time_steps,
            strike=strike,
            heston_v0=UNIT_MC_HESTON_V0,
            heston_theta=UNIT_MC_HESTON_THETA,
            heston_vol_of_vol=UNIT_MC_HESTON_VOL_OF_VOL,
        )
        milstein_price, milstein_se = calculate_heston_lsm_price(
            s0=UNIT_MC_SPOT,
            rate=UNIT_MC_RATE,
            n_paths=UNIT_TIME_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="milstein",
            correlation=correlation,
            time_steps=time_steps,
            strike=strike,
            heston_v0=UNIT_MC_HESTON_V0,
            heston_theta=UNIT_MC_HESTON_THETA,
            heston_vol_of_vol=UNIT_MC_HESTON_VOL_OF_VOL,
        )
        rows.append(
            [
                time_steps,
                euler_price,
                euler_se,
                milstein_price,
                milstein_se,
                reference_row[1],
                round(2.0 * reference_row[2], 8),
                reference_row[3],
                round(2.0 * reference_row[4], 8),
            ]
        )
    return rows


def build_unit_strike_time_convergence_sheet(
    *,
    sheet_name: str,
    sheet_title: str,
    correlation: float,
    reference_rows: list[list[Any]] | None = None,
) -> Sheet:
    headers = [
        "Number of time steps",
        "Heston American Put Price using Euler-Maruyama",
        "Euler-Maruyama s.e.",
        "Heston American Put Price using Milstein",
        "Milstein s.e.",
        "Euler reference price",
        "Euler reference +/- 2 s.e.",
        "Milstein reference price",
        "Milstein reference +/- 2 s.e.",
    ]
    reference_headers = [
        "Strike",
        "Euler reference price",
        "Euler reference s.e.",
        "Milstein reference price",
        "Milstein reference s.e.",
        "Milstein - Euler",
        "Combined s.e.",
        "+/- two combined s.e.",
        "Difference within +/- two combined s.e.",
    ]
    if reference_rows is None:
        reference_rows = calculate_reference_rows(correlation)
    rows: list[list[Any]] = [
        [sheet_title],
        [],
        ["Strike", ", ".join(f"{strike:g}" for strike in UNIT_MC_STRIKES)],
        ["Spot initial value", UNIT_MC_SPOT],
        ["Maturity", MATURITY],
        ["Rate", UNIT_MC_RATE],
        ["Dividend", DIVIDEND],
        ["Reference time steps", UNIT_REFERENCE_TIME_STEPS],
        ["Reference MC paths", f"{UNIT_REFERENCE_PATH_COUNT:,}"],
        ["Time-step convergence steps", ", ".join(str(step_count) for step_count in UNIT_TIME_STEP_COUNTS)],
        ["Time-step convergence MC paths", f"{UNIT_TIME_CONVERGENCE_PATH_COUNT:,}"],
        ["Polynomials", laguerre_polynomial_description(LAGUERRE_DEGREE)],
        ["Heston v0", UNIT_MC_HESTON_V0],
        ["Heston theta", UNIT_MC_HESTON_THETA],
        ["Heston rho", correlation],
        ["Heston kappa", HESTON_KAPPA],
        ["Heston vol of vol", UNIT_MC_HESTON_VOL_OF_VOL],
        ["Reference volatility", "sqrt(Heston theta)"],
        [],
        ["Reference price comparison"],
        reference_headers,
        *reference_rows,
        [],
    ]
    charts: list[ChartSpec] = []

    for strike_index, strike in enumerate(UNIT_MC_STRIKES):
        block_start = len(rows) + 1
        reference_row = reference_rows[strike_index]
        table_rows = calculate_time_convergence_rows_for_strike(
            strike=strike,
            strike_index=strike_index,
            correlation=correlation,
            reference_row=reference_row,
        )
        rows.append([f"Strike {strike:g}"])
        rows.append(headers)
        rows.extend(table_rows)
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + len(UNIT_TIME_STEP_COUNTS) - 1
        price_bounds = [
            value
            for row in table_rows
            for value in (
                row[1] - row[2],
                row[1] + row[2],
                row[3] - row[4],
                row[3] + row[4],
                row[5] - row[6],
                row[5] + row[6],
                row[7] - row[8],
                row[7] + row[8],
            )
        ]
        y_axis_min, y_axis_max = exact_bounds([float(value) for value in price_bounds])
        charts.append(
            (
                f"Milstein vs Euler time-step convergence, K={strike:g}",
                1 + strike_index * 18,
                10,
                [
                    (
                        "Euler-Maruyama",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    ),
                    (
                        "Milstein",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                    ),
                    (
                        "Euler reference",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                        f"'{sheet_name}'!$G${data_start}:$G${data_end}",
                        38100,
                    ),
                    (
                        "Milstein reference",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$H${data_start}:$H${data_end}",
                        f"'{sheet_name}'!$I${data_start}:$I${data_end}",
                        38100,
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Number of time steps",
                "American put price",
                "0",
                "0.000000",
            )
        )

    return (sheet_name, [], rows, charts)


def calculate_mc_path_convergence_rows_for_strike(
    *,
    strike: float,
    strike_index: int,
    correlation: float,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for path_index, n_paths in enumerate(UNIT_MC_CONVERGENCE_PATH_COUNTS):
        seed = (
            BASE_SEED
            + 17_000_000
            + int(UNIT_MC_RATE * 1_000_000)
            + int((correlation + 1.0) * 1_000_000)
            + strike_index * 10_000
            + path_index
        )
        print(
            f"Calculating MC convergence r={UNIT_MC_RATE:g}, corr={correlation:g}, "
            f"S0={UNIT_MC_SPOT:g}, K={strike:g}, steps={UNIT_MC_CONVERGENCE_TIME_STEPS:,}, "
            f"paths={n_paths:,}"
        )
        euler_price, euler_se = calculate_heston_lsm_price(
            s0=UNIT_MC_SPOT,
            rate=UNIT_MC_RATE,
            n_paths=n_paths,
            seed=seed,
            variance_scheme="euler",
            correlation=correlation,
            time_steps=UNIT_MC_CONVERGENCE_TIME_STEPS,
            strike=strike,
            heston_v0=UNIT_MC_HESTON_V0,
            heston_theta=UNIT_MC_HESTON_THETA,
            heston_vol_of_vol=UNIT_MC_HESTON_VOL_OF_VOL,
        )
        milstein_price, milstein_se = calculate_heston_lsm_price(
            s0=UNIT_MC_SPOT,
            rate=UNIT_MC_RATE,
            n_paths=n_paths,
            seed=seed,
            variance_scheme="milstein",
            correlation=correlation,
            time_steps=UNIT_MC_CONVERGENCE_TIME_STEPS,
            strike=strike,
            heston_v0=UNIT_MC_HESTON_V0,
            heston_theta=UNIT_MC_HESTON_THETA,
            heston_vol_of_vol=UNIT_MC_HESTON_VOL_OF_VOL,
        )
        rows.append([n_paths, euler_price, euler_se, milstein_price, milstein_se])
    return rows


def formula_cell(formula: str, cached_value: Any) -> tuple[str, str, Any]:
    return ("formula", formula, cached_value)


def build_unit_strike_mc_path_convergence_sheet(reference_rows: list[list[Any]]) -> Sheet:
    sheet_name = "Mlst vs Eul MC conv corr=0.3"
    source_sheet_name = "Mlst vs Eul corr=0.3"
    headers = [
        "Number of MC paths",
        "Heston American Put Price using Euler-Maruyama",
        "Euler-Maruyama s.e.",
        "Heston American Put Price using Milstein",
        "Milstein s.e.",
        "Euler reference price",
        "Euler reference s.e.",
        "Milstein reference price",
        "Milstein reference s.e.",
    ]
    rows: list[list[Any]] = [
        [sheet_name],
        [],
        ["Strike", ", ".join(f"{strike:g}" for strike in UNIT_MC_STRIKES)],
        ["Spot initial value", UNIT_MC_SPOT],
        ["Maturity", MATURITY],
        ["Rate", UNIT_MC_RATE],
        ["Dividend", DIVIDEND],
        ["Time-step convergence steps", UNIT_MC_CONVERGENCE_TIME_STEPS],
        ["Time-step convergence MC paths", ", ".join(f"{path_count:,}" for path_count in UNIT_MC_CONVERGENCE_PATH_COUNTS)],
        ["Reference price source sheet", source_sheet_name],
        ["Polynomials", laguerre_polynomial_description(LAGUERRE_DEGREE)],
        ["Heston v0", UNIT_MC_HESTON_V0],
        ["Heston theta", UNIT_MC_HESTON_THETA],
        ["Heston rho", UNIT_MC_CORRELATION],
        ["Heston kappa", HESTON_KAPPA],
        ["Heston vol of vol", UNIT_MC_HESTON_VOL_OF_VOL],
        ["Reference volatility", "sqrt(Heston theta)"],
        [],
    ]
    charts: list[ChartSpec] = []

    for strike_index, strike in enumerate(UNIT_MC_STRIKES):
        block_start = len(rows) + 1
        reference_row = reference_rows[strike_index]
        source_reference_row_number = 22 + strike_index
        euler_reference_formula = f"'{source_sheet_name}'!$B${source_reference_row_number}"
        euler_reference_se_formula = f"'{source_sheet_name}'!$C${source_reference_row_number}"
        milstein_reference_formula = f"'{source_sheet_name}'!$D${source_reference_row_number}"
        milstein_reference_se_formula = f"'{source_sheet_name}'!$E${source_reference_row_number}"
        table_rows = []
        for row in calculate_mc_path_convergence_rows_for_strike(
            strike=strike,
            strike_index=strike_index,
            correlation=UNIT_MC_CORRELATION,
        ):
            table_rows.append(
                row
                + [
                    formula_cell(euler_reference_formula, reference_row[1]),
                    formula_cell(f"2*{euler_reference_se_formula}", round(2.0 * reference_row[2], 8)),
                    formula_cell(milstein_reference_formula, reference_row[3]),
                    formula_cell(f"2*{milstein_reference_se_formula}", round(2.0 * reference_row[4], 8)),
                ]
            )

        rows.append([f"Strike {strike:g}"])
        rows.append(headers)
        rows.extend(table_rows)
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + len(UNIT_MC_CONVERGENCE_PATH_COUNTS) - 1
        price_bounds = [
            value
            for row in table_rows
            for value in (
                row[1] - row[2],
                row[1] + row[2],
                row[3] - row[4],
                row[3] + row[4],
                row[5][2] - row[6][2],
                row[5][2] + row[6][2],
                row[7][2] - row[8][2],
                row[7][2] + row[8][2],
            )
        ]
        y_axis_min, y_axis_max = exact_bounds([float(value) for value in price_bounds])
        charts.append(
            (
                f"Milstein vs Euler MC convergence, K={strike:g}",
                1 + strike_index * 18,
                10,
                [
                    (
                        "Euler-Maruyama",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    ),
                    (
                        "Milstein",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                    ),
                    (
                        "Euler reference",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                        f"'{sheet_name}'!$G${data_start}:$G${data_end}",
                        38100,
                    ),
                    (
                        "Milstein reference",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$H${data_start}:$H${data_end}",
                        f"'{sheet_name}'!$I${data_start}:$I${data_end}",
                        38100,
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Number of MC paths",
                "American put price",
                "0",
                "0.000000",
            )
        )

    return (sheet_name, [], rows, charts)


def calculate_average_variance_rows_for_spot(
    s0: float,
    correlation: float,
    spot_index: int,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    gbm_variance = round(HESTON_THETA * MATURITY, 8)

    for path_index, n_paths in enumerate(PATH_COUNTS):
        seed = (
            BASE_SEED
            + 9_000_000
            + int((correlation + 1.0) * 1_000_000)
            + spot_index * 10_000
            + path_index
        )
        print(f"Calculating average variance, corr={correlation:g}, S0={s0:g}, paths={n_paths:,}")
        euler_average_variance, euler_average_variance_se, euler_floor_hit_rate = calculate_average_heston_variance(
            s0=s0,
            rate=0.0,
            n_paths=n_paths,
            seed=seed,
            variance_scheme="euler",
            correlation=correlation,
        )
        milstein_average_variance, milstein_average_variance_se, milstein_floor_hit_rate = calculate_average_heston_variance(
            s0=s0,
            rate=0.0,
            n_paths=n_paths,
            seed=seed,
            variance_scheme="milstein",
            correlation=correlation,
        )
        rows.append(
            [
                n_paths,
                euler_average_variance,
                euler_average_variance_se,
                euler_floor_hit_rate,
                milstein_average_variance,
                milstein_average_variance_se,
                milstein_floor_hit_rate,
                gbm_variance,
            ]
        )

    return rows


def build_average_variance_sheet(
    correlation: float,
    *,
    sheet_name: str | None = None,
    sheet_title: str | None = None,
    strike: float = STRIKE,
    spot_levels: tuple[float, ...] = SPOT_LEVELS,
) -> Sheet:
    sheet_name = sheet_name or average_variance_sheet_name(correlation)
    headers = [
        "Number of MC paths",
        "Heston Euler-Maruyama average variance",
        "Heston Euler-Maruyama average variance s.e.",
        "Floor hit rate Eul-Mar",
        "Heston Milstein average variance",
        "Heston Milstein average variance s.e.",
        "Floor hit rate Milst",
        "GBM variance",
    ]
    rows = [[sheet_title], []] if sheet_title else []
    rows.extend(parameter_rows(0.0, correlation, strike=strike, spot_levels=spot_levels))
    charts: list[ChartSpec] = []
    spot_profile_rows: list[list[Any]] = [
        [
            "Spot",
            "Heston Euler-Maruyama average variance",
            "Heston Euler-Maruyama average variance s.e.",
            "Floor hit rate Eul-Mar",
            "Heston Milstein average variance",
            "Heston Milstein average variance s.e.",
            "Floor hit rate Milst",
            "GBM variance",
        ]
    ]

    for spot_index, s0 in enumerate(spot_levels):
        block_start = len(rows) + 1
        table_rows = calculate_average_variance_rows_for_spot(s0, correlation, spot_index)
        final_path_row = table_rows[-1]
        spot_profile_rows.append(
            [
                s0,
                final_path_row[1],
                final_path_row[2],
                final_path_row[3],
                final_path_row[4],
                final_path_row[5],
                final_path_row[6],
                final_path_row[7],
            ]
        )
        rows.append([f"Initial spot {s0:g}"])
        rows.append(headers)
        rows.extend(table_rows)
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + len(PATH_COUNTS) - 1
        values = [
            value
            for row in table_rows
            for value in (
                row[1] - row[2],
                row[1] + row[2],
                row[4] - row[5],
                row[4] + row[5],
                row[7],
            )
        ]
        y_axis_min, y_axis_max = padded_bounds(values)
        charts.append(
            (
                f"Average total variance, S0={s0:g}",
                1 + spot_index * 18,
                7,
                [
                    (
                        "Heston Euler-Maruyama",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    ),
                    (
                        "Heston Milstein",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                        f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                    ),
                    (
                        "GBM variance",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$H${data_start}:$H${data_end}",
                        "",
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Number of MC paths",
                "Average total variance",
                "0",
                "0.000000",
            )
        )

    profile_start = len(rows) + 1
    rows.append(["100,000 path average variance profile by spot"])
    rows.extend(spot_profile_rows)
    profile_header_row = profile_start + 1
    profile_data_start = profile_header_row + 1
    profile_data_end = profile_data_start + len(spot_levels) - 1
    profile_bounds = [
        value
        for row in spot_profile_rows[1:]
        for value in (
            row[1] - row[2],
            row[1] + row[2],
            row[4] - row[5],
            row[4] + row[5],
            row[7],
        )
    ]
    profile_y_axis_min, profile_y_axis_max = padded_bounds([float(value) for value in profile_bounds])
    charts.append(
        (
            "100,000 path average variance by spot",
            1,
            14,
            [
                (
                    "Heston Euler-Maruyama",
                    f"'{sheet_name}'!$A${profile_data_start}:$A${profile_data_end}",
                    f"'{sheet_name}'!$B${profile_data_start}:$B${profile_data_end}",
                    f"'{sheet_name}'!$C${profile_data_start}:$C${profile_data_end}",
                ),
                (
                    "Heston Milstein",
                    f"'{sheet_name}'!$A${profile_data_start}:$A${profile_data_end}",
                    f"'{sheet_name}'!$E${profile_data_start}:$E${profile_data_end}",
                    f"'{sheet_name}'!$F${profile_data_start}:$F${profile_data_end}",
                ),
                (
                    "GBM variance",
                    f"'{sheet_name}'!$A${profile_data_start}:$A${profile_data_end}",
                    f"'{sheet_name}'!$H${profile_data_start}:$H${profile_data_end}",
                    "",
                ),
            ],
            "line",
            profile_y_axis_min,
            profile_y_axis_max,
            "Spot",
            "Average total variance",
            "0.0",
            "0.000000",
        )
    )

    return (sheet_name, [], rows, charts)


def degree_convergence_parameter_rows(rate: float, correlation: float) -> list[list[Any]]:
    return [
        ["Strike", STRIKE],
        ["Maturity", MATURITY],
        ["Rate", rate],
        ["Dividend", DIVIDEND],
        ["Number of MC paths", f"{DEGREE_CONVERGENCE_PATH_COUNT:,}"],
        ["Polynomials tested", ", ".join(laguerre_polynomial_description(degree) for degree in LAGUERRE_DEGREES)],
        ["Number of time steps", NUMBER_OF_TIME_STEPS],
        ["Spot initial value", ", ".join(f"{spot:g}" for spot in SPOT_LEVELS)],
        ["Heston v0", HESTON_V0],
        ["Heston theta", HESTON_THETA],
        ["Heston rho", correlation],
        ["Heston kappa", HESTON_KAPPA],
        ["Heston vol of vol", HESTON_VOL_OF_VOL],
        ["Reference volatility", "sqrt(Heston theta)"],
        [],
    ]


def calculate_degree_convergence_rows_for_spot(
    s0: float,
    rate: float,
    correlation: float,
    spot_index: int,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    reference_sigma = sqrt(HESTON_THETA)
    baw_price = round(
        american_put_price_barone_adesi_whaley(
            s0,
            STRIKE,
            rate,
            reference_sigma,
            MATURITY,
            dividend_yield=DIVIDEND,
        ),
        8,
    )
    european_price = round(european_put_price(s0, STRIKE, rate, reference_sigma, MATURITY), 8)

    for degree_index, degree in enumerate(LAGUERRE_DEGREES):
        seed = (
            BASE_SEED
            + 11_000_000
            + int(rate * 1_000_000)
            + int((correlation + 1.0) * 1_000_000)
            + spot_index * 10_000
            + degree_index
        )
        print(
            f"Calculating degree convergence r={rate:g}, corr={correlation:g}, "
            f"S0={s0:g}, degree={degree}, paths={DEGREE_CONVERGENCE_PATH_COUNT:,}"
        )
        euler_price, euler_se = calculate_heston_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=DEGREE_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="euler",
            correlation=correlation,
            laguerre_degree=degree,
        )
        milstein_price, milstein_se = calculate_heston_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=DEGREE_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="milstein",
            correlation=correlation,
            laguerre_degree=degree,
        )
        gbm_price, gbm_se = calculate_gbm_lsm_price(
            s0=s0,
            rate=rate,
            n_paths=DEGREE_CONVERGENCE_PATH_COUNT,
            seed=seed,
            laguerre_degree=degree,
        )
        rows.append(
            [
                degree,
                euler_price,
                euler_se,
                milstein_price,
                milstein_se,
                gbm_price,
                gbm_se,
                baw_price,
                european_price,
            ]
        )
    return rows


def build_degree_convergence_sheet(rate: float = 0.0, correlation: float = 0.3) -> Sheet:
    sheet_name = "Base Mlst_Eul dg conv"
    headers = [
        "Polynomial degree",
        "Heston American Put Price using Euler-Maruyama",
        "Euler-Maruyama s.e.",
        "Heston American Put Price using Milstein",
        "Milstein s.e.",
        "GBM American Put Price",
        "GBM s.e.",
        "BAW Approx Price",
        "European Put Price",
    ]
    rows = [[f"Base Mlst_Eul degree convergence r={full_number(rate)} corr={full_number(correlation)}"], []]
    rows.extend(degree_convergence_parameter_rows(rate, correlation))
    charts: list[ChartSpec] = []

    for spot_index, s0 in enumerate(SPOT_LEVELS):
        block_start = len(rows) + 1
        table_rows = calculate_degree_convergence_rows_for_spot(s0, rate, correlation, spot_index)
        rows.append([f"Initial spot {s0:g}"])
        rows.append(headers)
        rows.extend(table_rows)
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + len(LAGUERRE_DEGREES) - 1
        price_bounds = [
            value
            for row in table_rows
            for value in (
                row[1] - row[2],
                row[1] + row[2],
                row[3] - row[4],
                row[3] + row[4],
                row[5] - row[6],
                row[5] + row[6],
                row[7],
                row[8],
            )
        ]
        y_axis_min, y_axis_max = padded_bounds([float(value) for value in price_bounds])
        charts.append(
            (
                f"Milstein vs Euler degree convergence, S0={s0:g}",
                1 + spot_index * 18,
                10,
                [
                    (
                        "Euler-Maruyama",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    ),
                    (
                        "Milstein",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$D${data_start}:$D${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                    ),
                    (
                        "GBM",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                        f"'{sheet_name}'!$G${data_start}:$G${data_end}",
                    ),
                    (
                        "BAW approximation",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$H${data_start}:$H${data_end}",
                        "",
                    ),
                    (
                        "European put",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$I${data_start}:$I${data_end}",
                        "",
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Polynomial degree",
                "American put price",
                "0",
                "0.000000",
            )
        )

    return (sheet_name, [], rows, charts)


def build_average_variance_degree_convergence_sheet(rate: float = 0.0, correlation: float = 0.3) -> Sheet:
    sheet_name = "Heston avg var dg conv"
    headers = [
        "Polynomial degree",
        "Heston Euler-Maruyama average variance",
        "Heston Euler-Maruyama average variance s.e.",
        "Floor hit rate Eul-Mar",
        "Heston Milstein average variance",
        "Heston Milstein average variance s.e.",
        "Floor hit rate Milst",
        "GBM variance",
    ]
    rows = [[f"Heston average variance degree convergence corr={full_number(correlation)}"], []]
    rows.extend(degree_convergence_parameter_rows(rate, correlation))
    charts: list[ChartSpec] = []

    for spot_index, s0 in enumerate(SPOT_LEVELS):
        block_start = len(rows) + 1
        seed = (
            BASE_SEED
            + 12_000_000
            + int((correlation + 1.0) * 1_000_000)
            + spot_index * 10_000
        )
        print(
            f"Calculating average variance degree convergence corr={correlation:g}, "
            f"S0={s0:g}, paths={DEGREE_CONVERGENCE_PATH_COUNT:,}"
        )
        euler_average_variance, euler_average_variance_se, euler_floor_hit_rate = calculate_average_heston_variance(
            s0=s0,
            rate=rate,
            n_paths=DEGREE_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="euler",
            correlation=correlation,
        )
        milstein_average_variance, milstein_average_variance_se, milstein_floor_hit_rate = calculate_average_heston_variance(
            s0=s0,
            rate=rate,
            n_paths=DEGREE_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="milstein",
            correlation=correlation,
        )
        gbm_variance = round(HESTON_THETA * MATURITY, 8)
        table_rows = [
            [
                degree,
                euler_average_variance,
                euler_average_variance_se,
                euler_floor_hit_rate,
                milstein_average_variance,
                milstein_average_variance_se,
                milstein_floor_hit_rate,
                gbm_variance,
            ]
            for degree in LAGUERRE_DEGREES
        ]
        rows.append([f"Initial spot {s0:g}"])
        rows.append(headers)
        rows.extend(table_rows)
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + len(LAGUERRE_DEGREES) - 1
        values = [
            value
            for row in table_rows
            for value in (
                row[1] - row[2],
                row[1] + row[2],
                row[4] - row[5],
                row[4] + row[5],
                row[7],
            )
        ]
        y_axis_min, y_axis_max = padded_bounds(values)
        charts.append(
            (
                f"Average total variance degree convergence, S0={s0:g}",
                1 + spot_index * 18,
                9,
                [
                    (
                        "Heston Euler-Maruyama",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$B${data_start}:$B${data_end}",
                        f"'{sheet_name}'!$C${data_start}:$C${data_end}",
                    ),
                    (
                        "Heston Milstein",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$E${data_start}:$E${data_end}",
                        f"'{sheet_name}'!$F${data_start}:$F${data_end}",
                    ),
                    (
                        "GBM variance",
                        f"'{sheet_name}'!$A${data_start}:$A${data_end}",
                        f"'{sheet_name}'!$H${data_start}:$H${data_end}",
                        "",
                    ),
                ],
                "line",
                y_axis_min,
                y_axis_max,
                "Polynomial degree",
                "Average total variance",
                "0",
                "0.000000",
            )
        )

    return (sheet_name, [], rows, charts)


def build_workbook_sheets() -> list[Sheet]:
    corr_0_3_reference_rows = calculate_reference_rows(0.3)
    return [
        build_unit_strike_mc_convergence_sheet(
            sheet_name="Mlst vs Eul corr=0.3",
            sheet_title="Mlst vs Eul corr=0.3",
            correlation=0.3,
            reference_rows=corr_0_3_reference_rows,
        ),
        build_unit_strike_mc_convergence_sheet(
            sheet_name="Mlst vs Eul corr=-0.7",
            sheet_title="Mlst vs Eul corr=-0.7",
            correlation=-0.7,
        ),
        build_unit_strike_mc_path_convergence_sheet(corr_0_3_reference_rows),
    ]


def main() -> None:
    write_workbook(DEFAULT_OUTPUT_PATH, build_workbook_sheets())
    print(f"Excel file saved to {DEFAULT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
