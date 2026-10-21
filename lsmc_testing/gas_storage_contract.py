"""Gas storage contract parameters from Boogert and de Jong Exhibit 1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class GasStorageContract:
    """Physical gas storage constraints in MWh."""

    start_date: date
    end_date: date
    min_volume: float
    max_volume: float
    start_volume: float
    end_volume: float
    max_withdrawal_per_day: float
    max_injection_per_day: float
    volume_grid_size: int | None = None

    def __post_init__(self) -> None:
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.min_volume >= self.max_volume:
            raise ValueError("min_volume must be less than max_volume")
        if not self.min_volume <= self.start_volume <= self.max_volume:
            raise ValueError("start_volume must lie inside storage bounds")
        if not self.min_volume <= self.end_volume <= self.max_volume:
            raise ValueError("end_volume must lie inside storage bounds")
        if self.max_withdrawal_per_day <= 0.0:
            raise ValueError("max_withdrawal_per_day must be positive")
        if self.max_injection_per_day <= 0.0:
            raise ValueError("max_injection_per_day must be positive")

        if self.volume_grid_size is None:
            grid_intervals = (self.max_volume - self.min_volume) / self.max_injection_per_day
            object.__setattr__(self, "volume_grid_size", int(round(grid_intervals)) + 1)

        if self.volume_grid_size < 2:
            raise ValueError("volume_grid_size must be at least 2")

        if not _lies_on_grid(self.min_volume, self.volume_step, self.start_volume):
            raise ValueError("start_volume must lie on the configured volume grid")
        if not _lies_on_grid(self.min_volume, self.volume_step, self.end_volume):
            raise ValueError("end_volume must lie on the configured volume grid")

    @property
    def n_days(self) -> int:
        return (self.end_date - self.start_date).days

    @property
    def dates(self) -> list[date]:
        return [self.start_date + timedelta(days=index) for index in range(self.n_days + 1)]

    @property
    def volume_step(self) -> float:
        return (self.max_volume - self.min_volume) / (self.volume_grid_size - 1)

    @property
    def volume_grid(self) -> list[float]:
        return [self.min_volume + index * self.volume_step for index in range(self.volume_grid_size)]

    def feasible_next_volumes(self, current_volume: float, days_remaining_after_step: int) -> list[float]:
        """Return grid volumes reachable tomorrow and still able to end at target volume."""
        min_next_by_rate = max(self.min_volume, current_volume - self.max_withdrawal_per_day)
        max_next_by_rate = min(self.max_volume, current_volume + self.max_injection_per_day)

        # Keep only volumes from which the terminal volume remains reachable.
        min_next_by_terminal = self.end_volume - days_remaining_after_step * self.max_injection_per_day
        max_next_by_terminal = self.end_volume + days_remaining_after_step * self.max_withdrawal_per_day
        lower = max(min_next_by_rate, min_next_by_terminal)
        upper = min(max_next_by_rate, max_next_by_terminal)

        return [volume for volume in self.volume_grid if lower - 1e-9 <= volume <= upper + 1e-9]


def exhibit1_contract(*, start_date: date = date(2005, 7, 1)) -> GasStorageContract:
    """Return the standard storage contract in Exhibit 1.

    The published table uses 07/01/2005 to 06/30/2006. Pass
    ``start_date=date(2005, 7, 7)`` if you want the literal 7 July start.
    """

    return GasStorageContract(
        start_date=start_date,
        end_date=date(start_date.year + 1, 6, 30),
        min_volume=0.0,
        max_volume=250_000.0,
        start_volume=100_000.0,
        end_volume=100_000.0,
        max_withdrawal_per_day=7_500.0,
        max_injection_per_day=2_500.0,
    )


def _lies_on_grid(grid_minimum: float, grid_step: float, value: float) -> bool:
    grid_units = (value - grid_minimum) / grid_step
    return abs(grid_units - round(grid_units)) <= 1e-9
