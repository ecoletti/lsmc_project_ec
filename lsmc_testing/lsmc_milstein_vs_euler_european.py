"""Compare Heston European put convergence under Euler-Maruyama and Milstein."""

from __future__ import annotations

from math import exp, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from lsmc_testing.excel import ChartSpec, Sheet, write_workbook
from lsmc_testing.lsmc_milstein_vs_euler import (
    BASE_SEED,
    DIVIDEND,
    HESTON_KAPPA,
    MATURITY,
    UNIT_MC_HESTON_THETA,
    UNIT_MC_HESTON_V0,
    UNIT_MC_HESTON_VOL_OF_VOL,
    UNIT_MC_RATE,
    UNIT_MC_SPOT,
    UNIT_MC_STRIKES,
    exact_bounds,
)
from lsmc_testing.simulation import simulate_heston_spot_scaled_vol_of_vol_paths


UNIT_REFERENCE_PATH_COUNT = 200_000
UNIT_REFERENCE_TIME_STEPS = 500
UNIT_TIME_STEP_COUNTS = (1, 3, 6, 12, 25, 50, 100, 150, 200, 250, 300, 350, 400)
UNIT_TIME_CONVERGENCE_PATH_COUNT = 100_000
UNIT_MC_CONVERGENCE_TIME_STEPS = 250
UNIT_MC_CONVERGENCE_PATH_COUNTS = (1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000)
DEFAULT_OUTPUT_PATH = Path("output") / "lsmc_milstein_vs_euler_european.xlsx"


def calculate_heston_european_put_price(
    *,
    strike: float,
    correlation: float,
    n_paths: int,
    seed: int,
    variance_scheme: str,
    time_steps: int,
) -> tuple[float, float]:
    paths = simulate_heston_spot_scaled_vol_of_vol_paths(
        s0=UNIT_MC_SPOT,
        rate=UNIT_MC_RATE,
        v0=UNIT_MC_HESTON_V0,
        kappa=HESTON_KAPPA,
        theta=UNIT_MC_HESTON_THETA,
        vol_of_vol=UNIT_MC_HESTON_VOL_OF_VOL,
        rho=correlation,
        maturity=MATURITY,
        steps_per_year=time_steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=True,
        variance_scheme=variance_scheme,
    )
    discounted_payoffs = exp(-UNIT_MC_RATE * MATURITY) * np.maximum(strike - paths[-1], 0.0)
    return (
        round(float(discounted_payoffs.mean()), 8),
        round(float(discounted_payoffs.std(ddof=1) / sqrt(n_paths)), 8),
    )


def calculate_reference_rows_for_strike(
    *,
    strike: float,
    strike_index: int,
    correlation: float,
) -> list[Any]:
    seed = (
        BASE_SEED
        + 25_000_000
        + int(UNIT_MC_RATE * 1_000_000)
        + int((correlation + 1.0) * 1_000_000)
        + strike_index * 10_000
    )
    print(
        f"Calculating European reference r={UNIT_MC_RATE:g}, corr={correlation:g}, "
        f"S0={UNIT_MC_SPOT:g}, K={strike:g}, steps={UNIT_REFERENCE_TIME_STEPS:,}, "
        f"paths={UNIT_REFERENCE_PATH_COUNT:,}"
    )
    euler_price, euler_se = calculate_heston_european_put_price(
        strike=strike,
        correlation=correlation,
        n_paths=UNIT_REFERENCE_PATH_COUNT,
        seed=seed,
        variance_scheme="euler",
        time_steps=UNIT_REFERENCE_TIME_STEPS,
    )
    milstein_price, milstein_se = calculate_heston_european_put_price(
        strike=strike,
        correlation=correlation,
        n_paths=UNIT_REFERENCE_PATH_COUNT,
        seed=seed,
        variance_scheme="milstein",
        time_steps=UNIT_REFERENCE_TIME_STEPS,
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
    seed = (
        BASE_SEED
        + 26_000_000
        + int(UNIT_MC_RATE * 1_000_000)
        + int((correlation + 1.0) * 1_000_000)
        + strike_index * 10_000
    )
    for time_steps in UNIT_TIME_STEP_COUNTS:
        print(
            f"Calculating European time-step convergence r={UNIT_MC_RATE:g}, corr={correlation:g}, "
            f"S0={UNIT_MC_SPOT:g}, K={strike:g}, steps={time_steps:,}, "
            f"paths={UNIT_TIME_CONVERGENCE_PATH_COUNT:,}"
        )
        euler_price, euler_se = calculate_heston_european_put_price(
            strike=strike,
            correlation=correlation,
            n_paths=UNIT_TIME_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="euler",
            time_steps=time_steps,
        )
        milstein_price, milstein_se = calculate_heston_european_put_price(
            strike=strike,
            correlation=correlation,
            n_paths=UNIT_TIME_CONVERGENCE_PATH_COUNT,
            seed=seed,
            variance_scheme="milstein",
            time_steps=time_steps,
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
            + 27_000_000
            + int(UNIT_MC_RATE * 1_000_000)
            + int((correlation + 1.0) * 1_000_000)
            + strike_index * 10_000
            + path_index
        )
        print(
            f"Calculating European MC convergence r={UNIT_MC_RATE:g}, corr={correlation:g}, "
            f"S0={UNIT_MC_SPOT:g}, K={strike:g}, steps={UNIT_MC_CONVERGENCE_TIME_STEPS:,}, "
            f"paths={n_paths:,}"
        )
        euler_price, euler_se = calculate_heston_european_put_price(
            strike=strike,
            correlation=correlation,
            n_paths=n_paths,
            seed=seed,
            variance_scheme="euler",
            time_steps=UNIT_MC_CONVERGENCE_TIME_STEPS,
        )
        milstein_price, milstein_se = calculate_heston_european_put_price(
            strike=strike,
            correlation=correlation,
            n_paths=n_paths,
            seed=seed,
            variance_scheme="milstein",
            time_steps=UNIT_MC_CONVERGENCE_TIME_STEPS,
        )
        rows.append([n_paths, euler_price, euler_se, milstein_price, milstein_se])
    return rows


def workbook_parameter_rows(correlation: float) -> list[list[Any]]:
    return [
        ["Strike", ", ".join(f"{strike:g}" for strike in UNIT_MC_STRIKES)],
        ["Spot initial value", UNIT_MC_SPOT],
        ["Maturity", MATURITY],
        ["Rate", UNIT_MC_RATE],
        ["Dividend", DIVIDEND],
        ["Reference time steps", UNIT_REFERENCE_TIME_STEPS],
        ["Reference MC paths", f"{UNIT_REFERENCE_PATH_COUNT:,}"],
        ["Time-step convergence steps", ", ".join(str(step_count) for step_count in UNIT_TIME_STEP_COUNTS)],
        ["Time-step convergence MC paths", f"{UNIT_TIME_CONVERGENCE_PATH_COUNT:,}"],
        ["Payoff", "European put terminal payoff"],
        ["Heston v0", UNIT_MC_HESTON_V0],
        ["Heston theta", UNIT_MC_HESTON_THETA],
        ["Heston rho", correlation],
        ["Heston kappa", HESTON_KAPPA],
        ["Heston vol of vol", UNIT_MC_HESTON_VOL_OF_VOL],
        ["Reference volatility", "sqrt(Heston theta)"],
    ]


def reference_headers() -> list[str]:
    return [
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


def build_time_convergence_sheet(
    *,
    sheet_name: str,
    sheet_title: str,
    correlation: float,
    reference_rows: list[list[Any]] | None = None,
) -> Sheet:
    headers = [
        "Number of time steps",
        "Heston European Put Price using Euler-Maruyama",
        "Euler-Maruyama s.e.",
        "Heston European Put Price using Milstein",
        "Milstein s.e.",
        "Euler reference price",
        "Euler reference +/- 2 s.e.",
        "Milstein reference price",
        "Milstein reference +/- 2 s.e.",
    ]
    if reference_rows is None:
        reference_rows = calculate_reference_rows(correlation)
    rows: list[list[Any]] = [
        [sheet_title],
        [],
        *workbook_parameter_rows(correlation),
        [],
        ["Reference price comparison"],
        reference_headers(),
        *reference_rows,
        [],
    ]
    charts: list[ChartSpec] = []

    for strike_index, strike in enumerate(UNIT_MC_STRIKES):
        block_start = len(rows) + 1
        table_rows = calculate_time_convergence_rows_for_strike(
            strike=strike,
            strike_index=strike_index,
            correlation=correlation,
            reference_row=reference_rows[strike_index],
        )
        rows.append([f"Strike {strike:g}"])
        rows.append(headers)
        rows.extend(table_rows)
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + len(UNIT_TIME_STEP_COUNTS) - 1
        y_axis_min, y_axis_max = _price_bounds(table_rows)
        charts.append(_convergence_chart(sheet_name, strike, data_start, data_end, y_axis_min, y_axis_max, "time-step"))

    return (sheet_name, [], rows, charts)


def build_mc_path_convergence_sheet(reference_rows: list[list[Any]]) -> Sheet:
    sheet_name = "Mlst vs Eul MC conv corr=0.3"
    source_sheet_name = "Mlst vs Eul corr=0.3"
    headers = [
        "Number of MC paths",
        "Heston European Put Price using Euler-Maruyama",
        "Euler-Maruyama s.e.",
        "Heston European Put Price using Milstein",
        "Milstein s.e.",
        "Euler reference price",
        "Euler reference +/- 2 s.e.",
        "Milstein reference price",
        "Milstein reference +/- 2 s.e.",
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
        ["Payoff", "European put terminal payoff"],
        ["Heston v0", UNIT_MC_HESTON_V0],
        ["Heston theta", UNIT_MC_HESTON_THETA],
        ["Heston rho", 0.3],
        ["Heston kappa", HESTON_KAPPA],
        ["Heston vol of vol", UNIT_MC_HESTON_VOL_OF_VOL],
        ["Reference volatility", "sqrt(Heston theta)"],
        [],
    ]
    charts: list[ChartSpec] = []

    for strike_index, strike in enumerate(UNIT_MC_STRIKES):
        block_start = len(rows) + 1
        reference_row = reference_rows[strike_index]
        table_rows = []
        for row in calculate_mc_path_convergence_rows_for_strike(
            strike=strike,
            strike_index=strike_index,
            correlation=0.3,
        ):
            table_rows.append(
                row
                + [
                    reference_row[1],
                    round(2.0 * reference_row[2], 8),
                    reference_row[3],
                    round(2.0 * reference_row[4], 8),
                ]
            )

        rows.append([f"Strike {strike:g}"])
        rows.append(headers)
        rows.extend(table_rows)
        rows.append([])

        data_start = block_start + 2
        data_end = data_start + len(UNIT_MC_CONVERGENCE_PATH_COUNTS) - 1
        y_axis_min, y_axis_max = _price_bounds(
            [
                [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                    row[8],
                ]
                for row in table_rows
            ]
        )
        charts.append(_convergence_chart(sheet_name, strike, data_start, data_end, y_axis_min, y_axis_max, "MC"))

    return (sheet_name, [], rows, charts)


def _price_bounds(table_rows: list[list[Any]]) -> tuple[float, float]:
    values = [
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
    return exact_bounds([float(value) for value in values])


def _convergence_chart(
    sheet_name: str,
    strike: float,
    data_start: int,
    data_end: int,
    y_axis_min: float,
    y_axis_max: float,
    convergence_type: str,
) -> ChartSpec:
    x_axis_title = "Number of MC paths" if convergence_type == "MC" else "Number of time steps"
    title = (
        f"Milstein vs Euler MC convergence, K={strike:g}"
        if convergence_type == "MC"
        else f"Milstein vs Euler time-step convergence, K={strike:g}"
    )
    return (
        title,
        1 + UNIT_MC_STRIKES.index(strike) * 18,
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
        x_axis_title,
        "European put price",
        "0",
        "0.000000",
    )


def build_workbook_sheets() -> list[Sheet]:
    corr_0_3_reference_rows = calculate_reference_rows(0.3)
    return [
        build_time_convergence_sheet(
            sheet_name="Mlst vs Eul corr=0.3",
            sheet_title="Mlst vs Eul corr=0.3",
            correlation=0.3,
            reference_rows=corr_0_3_reference_rows,
        ),
        build_time_convergence_sheet(
            sheet_name="Mlst vs Eul corr=-0.7",
            sheet_title="Mlst vs Eul corr=-0.7",
            correlation=-0.7,
        ),
        build_mc_path_convergence_sheet(corr_0_3_reference_rows),
    ]


def main() -> None:
    write_workbook(DEFAULT_OUTPUT_PATH, build_workbook_sheets())
    print(f"Excel file saved to {DEFAULT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
