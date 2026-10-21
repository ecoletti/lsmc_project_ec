"""Generate Boogert-de Jong Exhibit-2 style gas storage volume bands."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np

from lsmc_testing.gas_storage_contract import exhibit1_contract
from lsmc_testing.gas_storage_simulation import GasPriceSimulationSpec, constant_mu, simulate_schwartz_spot_paths
from lsmc_testing.gas_storage_valuation import intrinsic_schedule, trigger_policy_volumes, volume_band


DEFAULT_OUTPUT_PATH = Path("output") / "gas_storage_exhibit2_volume_bands.csv"


def build_exhibit2_rows(
    *,
    n_paths: int = 10_000,
    initial_price: float = 20.0,
    mu_level: float = 20.0,
    kappa: float = 0.05,
    sigma: float = 0.0945,
    seed: int = 20261002,
    start_day: int = 1,
) -> list[list[object]]:
    """Return rows with intrinsic and simulated extrinsic volume bands.

    ``start_day=1`` means July 1, matching Exhibit 1. Use ``start_day=7`` for
    a literal July 7 start.
    """

    contract = exhibit1_contract(start_date=date(2005, 7, start_day))
    spec = GasPriceSimulationSpec(
        n_paths=n_paths,
        n_days=contract.n_days,
        initial_price=initial_price,
        kappa=kappa,
        sigma=sigma,
        seed=seed,
        antithetic=True,
    )
    paths = simulate_schwartz_spot_paths(spec, mu=constant_mu(mu_level))
    intrinsic_volumes = intrinsic_schedule(paths.mean(axis=1), contract)
    extrinsic_volumes = trigger_policy_volumes(paths, contract)
    intrinsic = volume_band(intrinsic_volumes)
    extrinsic = volume_band(extrinsic_volumes)

    rows: list[list[object]] = [
        [
            "day",
            "date",
            "intrinsic_min_volume",
            "intrinsic_max_volume",
            "extrinsic_min_volume",
            "extrinsic_max_volume",
            "spot_min",
            "spot_mean",
            "spot_max",
        ]
    ]
    for day, current_date in enumerate(contract.dates):
        rows.append(
            [
                day,
                current_date.isoformat(),
                round(float(intrinsic.minimum[day]), 6),
                round(float(intrinsic.maximum[day]), 6),
                round(float(extrinsic.minimum[day]), 6),
                round(float(extrinsic.maximum[day]), 6),
                round(float(paths[day].min()), 6),
                round(float(paths[day].mean()), 6),
                round(float(paths[day].max()), 6),
            ]
        )
    return rows


def write_rows(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-paths", type=int, default=10_000)
    parser.add_argument("--initial-price", type=float, default=20.0)
    parser.add_argument("--mu-level", type=float, default=20.0)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.0945)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--start-day", type=int, default=1, choices=range(1, 32))
    args = parser.parse_args()

    rows = build_exhibit2_rows(
        n_paths=args.n_paths,
        initial_price=args.initial_price,
        mu_level=args.mu_level,
        kappa=args.kappa,
        sigma=args.sigma,
        seed=args.seed,
        start_day=args.start_day,
    )
    write_rows(args.output, rows)
    print(f"CSV saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
