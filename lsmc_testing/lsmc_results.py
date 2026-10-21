"""Command-line runner for the Longstaff-Schwartz Table 1 benchmark."""

from __future__ import annotations

from pathlib import Path

from typing import Any

from lsmc_testing.benchmark import TABLE1_CASES, TableCase
from lsmc_testing.black_scholes import european_put_price
from lsmc_testing.excel import ChartSpec, Sheet, write_workbook
from lsmc_testing.pricing import price_american_put_lsm
from lsmc_testing.simulation import simulate_gbm_paths


STRIKE = 40.0
RATE = 0.06
NUMBER_PATHS = 100_000
BASE_SEED = 12345
VALUATION_SEED_OFFSET = 1_000_000
TABLE1_STEPS_PER_YEAR = 50
STEP_SENSITIVITY_VALUES = range(50, 351, 50)
SENSITIVITY_VOLATILITIES = (0.20, 0.30, 0.40, 0.50, 0.60)
DEFAULT_OUTPUT_PATH = Path("output") / "longstaff_schwartz_gbm_american_put_out_of_sample.xlsx"
CONVERGENCE_BIAS_S0 = 40.0
CONVERGENCE_BIAS_SIGMA = 0.40
CONVERGENCE_BIAS_MATURITY = 1.0
CONVERGENCE_BIAS_STEPS_PER_YEAR = 350
CONVERGENCE_BIAS_PATH_COUNTS = tuple(500 * 2**index for index in range(10))
CONVERGENCE_BIAS_SHEET_NAME = "Convergence bias S0 40 T 1"
CONFIDENCE_INTERVAL_MULTIPLIER = 1.96
PAPER_CASE_KEYS = {
    (case.s0, case.sigma, case.maturity)
    for case in TABLE1_CASES
}
HEADERS = [
    "S0",
    "Sigma",
    "T",
    "LSM American",
    "LSM s.e.",
    "Paper American",
    "Paper s.e.",
    "European BS",
    "Early Exercise",
    "Diff with paper",
]
def build_sensitivity_cases() -> tuple[TableCase, ...]:
    paper_lookup = {
        (case.s0, case.sigma, case.maturity): case
        for case in TABLE1_CASES
    }
    s0_values = sorted({case.s0 for case in TABLE1_CASES})
    maturity_values = sorted({case.maturity for case in TABLE1_CASES})
    cases = []

    for sigma in SENSITIVITY_VOLATILITIES:
        for s0 in s0_values:
            for maturity in maturity_values:
                paper_case = paper_lookup.get((s0, sigma, maturity))
                if paper_case is not None:
                    cases.append(paper_case)
                else:
                    cases.append(TableCase(s0, sigma, maturity, 0.0, 0.0, 0.0))

    return tuple(sorted(cases, key=lambda case: (case.sigma, case.s0, case.maturity)))


def calculate_rows(
    steps_per_year: int,
    number_paths: int,
    seed: int,
    table_cases: tuple[TableCase, ...] = TABLE1_CASES,
) -> list[list[Any]]:
    rows_by_key = {}
    group_keys = {
        (case.sigma, case.maturity)
        for case in table_cases
    }
    group_seed_offsets = {
        key: index
        for index, key in enumerate(sorted(group_keys), start=1)
    }
    cases_by_group = {
        group_key: [
            case for case in table_cases
            if (case.sigma, case.maturity) == group_key
        ]
        for group_key in group_keys
    }

    for group_key in sorted(group_keys):
        sigma, maturity = group_key
        unit_paths = simulate_gbm_paths(
            s0=1.0,
            rate=RATE,
            sigma=sigma,
            maturity=maturity,
            steps_per_year=steps_per_year,
            n_paths=number_paths,
            seed=seed + group_seed_offsets[group_key],
            antithetic=True,
        )
        valuation_unit_paths = simulate_gbm_paths(
            s0=1.0,
            rate=RATE,
            sigma=sigma,
            maturity=maturity,
            steps_per_year=steps_per_year,
            n_paths=number_paths,
            seed=seed + VALUATION_SEED_OFFSET + group_seed_offsets[group_key],
            antithetic=True,
        )
        for case in cases_by_group[group_key]:
            paths = unit_paths * case.s0
            valuation_paths = valuation_unit_paths * case.s0
            result = price_american_put_lsm(
                paths,
                strike=STRIKE,
                rate=RATE,
                maturity=case.maturity,
                valuation_paths=valuation_paths,
            )
            european = european_put_price(case.s0, STRIKE, RATE, case.sigma, case.maturity)
            early_exercise = result.price - european
            has_paper_value = (case.s0, case.sigma, case.maturity) in PAPER_CASE_KEYS
            paper_american = case.paper_simulated_american if has_paper_value else ""
            paper_standard_error = case.paper_standard_error if has_paper_value else ""
            diff = round(result.price - case.paper_simulated_american, 6) if has_paper_value else ""
            rows_by_key[(case.s0, case.sigma, case.maturity)] = [
                case.s0,
                case.sigma,
                case.maturity,
                round(result.price, 6),
                round(result.standard_error, 6),
                paper_american,
                paper_standard_error,
                round(european, 6),
                round(early_exercise, 6),
                diff,
            ]

    return [
        rows_by_key[(case.s0, case.sigma, case.maturity)]
        for case in table_cases
    ]


def print_rows(rows: list[list[Any]]) -> None:
    print(
        " S0 sigma   T |  LSM Am  s.e. | Paper Am Paper s.e. | Euro BS EarlyEx | Diff with paper"
    )
    print("-" * 83)
    for row in rows:
        print(
            f"{row[0]:3.0f} {row[1]:5.2f} {row[2]:3.0f} |"
            f" {row[3]:7.3f} {row[4]:5.3f} |"
            f" {row[5]:8.3f} {row[6]:9.3f} |"
            f" {row[7]:7.3f} {row[8]:7.3f} |"
            f" {row[9]:+6.3f}"
        )


def build_convergence_sheets(sensitivity_results: dict[int, list[list[Any]]]) -> list[Sheet]:
    sheets = []
    s0_values = sorted({case.s0 for case in TABLE1_CASES})
    maturity_values = sorted({case.maturity for case in TABLE1_CASES})

    for s0 in s0_values:
        for maturity in maturity_values:
            sheet_name = f"S0 {s0:.0f} T {maturity:.0f}"
            sheet_ref = f"'{sheet_name}'"
            rows: list[list[Any]] = []
            charts: list[ChartSpec] = []
            title = f"S0={s0:.0f}, T={maturity:.0f}"
            rows.append([title])
            header = ["STEPS_PER_YEAR"]
            for sigma in SENSITIVITY_VOLATILITIES:
                header.extend([f"Sigma {sigma:.2f} LSM American", f"Sigma {sigma:.2f} LSM s.e."])
            rows.append(header)

            for steps_per_year, result_rows in sensitivity_results.items():
                values = [steps_per_year]
                for sigma in SENSITIVITY_VOLATILITIES:
                    row = next(
                        item for item in result_rows
                        if item[0] == s0 and item[1] == sigma and item[2] == maturity
                    )
                    values.extend([row[3], row[4]])
                rows.append(values)

            data_start = 3
            data_end = data_start + len(sensitivity_results) - 1
            for index, sigma in enumerate(SENSITIVITY_VOLATILITIES):
                value_column = 2 + index * 2
                error_column = value_column + 1
                chart_row = 1 + index * 16
                chart_column = 14
                charts.append(
                    (
                        f"LSM American convergence, {title}, sigma={sigma:.2f}",
                        chart_row,
                        chart_column,
                        [
                            (
                                f"Sigma {sigma:.2f}",
                                f"{sheet_ref}!$A${data_start}:$A${data_end}",
                                f"{sheet_ref}!${_column_name(value_column)}${data_start}:${_column_name(value_column)}${data_end}",
                                f"{sheet_ref}!${_column_name(error_column)}${data_start}:${_column_name(error_column)}${data_end}",
                            )
                        ],
                    )
                )

            sheets.append((sheet_name, [], rows, charts))

    return sheets


def build_convergence_bias_sheet(seed: int = BASE_SEED) -> Sheet:
    rows: list[list[Any]] = []
    for index, number_paths in enumerate(CONVERGENCE_BIAS_PATH_COUNTS, start=1):
        training_paths = simulate_gbm_paths(
            s0=CONVERGENCE_BIAS_S0,
            rate=RATE,
            sigma=CONVERGENCE_BIAS_SIGMA,
            maturity=CONVERGENCE_BIAS_MATURITY,
            steps_per_year=CONVERGENCE_BIAS_STEPS_PER_YEAR,
            n_paths=number_paths,
            seed=seed + index,
            antithetic=True,
        )
        valuation_paths = simulate_gbm_paths(
            s0=CONVERGENCE_BIAS_S0,
            rate=RATE,
            sigma=CONVERGENCE_BIAS_SIGMA,
            maturity=CONVERGENCE_BIAS_MATURITY,
            steps_per_year=CONVERGENCE_BIAS_STEPS_PER_YEAR,
            n_paths=number_paths,
            seed=seed + VALUATION_SEED_OFFSET + index,
            antithetic=True,
        )
        in_sample = price_american_put_lsm(
            training_paths,
            strike=STRIKE,
            rate=RATE,
            maturity=CONVERGENCE_BIAS_MATURITY,
        )
        out_of_sample = price_american_put_lsm(
            training_paths,
            strike=STRIKE,
            rate=RATE,
            maturity=CONVERGENCE_BIAS_MATURITY,
            valuation_paths=valuation_paths,
        )
        rows.append(
            [
                number_paths,
                round(in_sample.price, 6),
                round(in_sample.standard_error, 6),
                round(CONFIDENCE_INTERVAL_MULTIPLIER * in_sample.standard_error, 6),
                round(out_of_sample.price, 6),
                round(out_of_sample.standard_error, 6),
                round(CONFIDENCE_INTERVAL_MULTIPLIER * out_of_sample.standard_error, 6),
            ]
        )

    data_start = 2
    data_end = data_start + len(rows) - 1
    sheet_ref = f"'{CONVERGENCE_BIAS_SHEET_NAME}'"
    chart = (
        "American put convergence bias, S0=40, sigma=0.40, T=1, steps/year=350",
        1,
        9,
        [
            (
                "In-sample",
                f"{sheet_ref}!$A${data_start}:$A${data_end}",
                f"{sheet_ref}!$B${data_start}:$B${data_end}",
                f"{sheet_ref}!$D${data_start}:$D${data_end}",
            ),
            (
                "Out-of-sample",
                f"{sheet_ref}!$A${data_start}:$A${data_end}",
                f"{sheet_ref}!$E${data_start}:$E${data_end}",
                f"{sheet_ref}!$G${data_start}:$G${data_end}",
            ),
        ],
        "scatter",
        None,
        None,
        "MC paths",
        "American put price",
        None,
        None,
        "0",
        "0.000",
        True,
        {
            "scatter_style": "lineMarker",
            "text_boxes": [
                {
                    "row": 20,
                    "column": 9,
                    "to_row": 30,
                    "to_column": 19,
                    "name": "Confidence interval calculation",
                    "text": (
                        "Confidence interval calculation\n"
                        "For each MC path count, the standard error is calculated as "
                        "sample_std(present_values, ddof=1) / sqrt(number_paths).\n"
                        "The chart error bars show a two-sided 95% normal confidence interval: "
                        "LSM American price +/- 1.96 * standard error.\n"
                        "Equivalently, lower = price - 1.96 * s.e. and upper = price + 1.96 * s.e.\n"
                        "In-sample values use the same simulated paths to fit and value the exercise rule. "
                        "Out-of-sample values fit the exercise rule on training paths and value it on an "
                        "independent path set."
                    ),
                }
            ],
        },
    )
    return (
        CONVERGENCE_BIAS_SHEET_NAME,
        [
            "MC Paths",
            "In-sample LSM American",
            "In-sample s.e.",
            "In-sample 95% CI half-width",
            "Out-of-sample LSM American",
            "Out-of-sample s.e.",
            "Out-of-sample 95% CI half-width",
        ],
        rows,
        [chart],
    )


def _column_name(column_index: int) -> str:
    name = ""
    while column_index:
        column_index, remainder = divmod(column_index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def main() -> None:
    sheets: list[Sheet] = []
    sensitivity_cases = build_sensitivity_cases()
    sensitivity_results: dict[int, list[list[Any]]] = {}

    print(f"Calculating Table 1 replication with {NUMBER_PATHS} training paths and {NUMBER_PATHS} valuation paths...")
    table1_rows = calculate_rows(TABLE1_STEPS_PER_YEAR, NUMBER_PATHS, BASE_SEED)
    print_rows(table1_rows)
    sheets.append(("Table 1 Replication", HEADERS, table1_rows))

    for steps_per_year in STEP_SENSITIVITY_VALUES:
        print(f"\nCalculating STEPS_PER_YEAR={steps_per_year} with {NUMBER_PATHS} training paths and {NUMBER_PATHS} valuation paths...")
        rows = calculate_rows(steps_per_year, NUMBER_PATHS, BASE_SEED, sensitivity_cases)
        sensitivity_results[steps_per_year] = rows
        sheets.append(('Discretization {0}'.format(steps_per_year), HEADERS, rows))

    sheets.extend(build_convergence_sheets(sensitivity_results))
    print(f"\nCalculating {CONVERGENCE_BIAS_SHEET_NAME}...")
    sheets.append(build_convergence_bias_sheet(BASE_SEED))
    write_workbook(DEFAULT_OUTPUT_PATH, sheets)
    print(f"\nExcel file saved to {DEFAULT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
