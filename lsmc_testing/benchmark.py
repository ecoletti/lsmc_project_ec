"""Benchmark cases and published values for Table 1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableCase:
    s0: float
    sigma: float
    maturity: float
    finite_difference_american: float
    paper_simulated_american: float
    paper_standard_error: float


TABLE1_CASES: tuple[TableCase, ...] = (
    TableCase(36, 0.20, 1, 4.478, 4.472, 0.010),
    TableCase(36, 0.20, 2, 4.840, 4.821, 0.012),
    TableCase(36, 0.40, 1, 7.101, 7.091, 0.020),
    TableCase(36, 0.40, 2, 8.508, 8.488, 0.024),
    TableCase(38, 0.20, 1, 3.250, 3.244, 0.009),
    TableCase(38, 0.20, 2, 3.745, 3.735, 0.011),
    TableCase(38, 0.40, 1, 6.148, 6.139, 0.019),
    TableCase(38, 0.40, 2, 7.670, 7.669, 0.022),
    TableCase(40, 0.20, 1, 2.314, 2.313, 0.009),
    TableCase(40, 0.20, 2, 2.885, 2.879, 0.010),
    TableCase(40, 0.40, 1, 5.312, 5.308, 0.018),
    TableCase(40, 0.40, 2, 6.920, 6.921, 0.022),
    TableCase(42, 0.20, 1, 1.617, 1.617, 0.007),
    TableCase(42, 0.20, 2, 2.212, 2.206, 0.010),
    TableCase(42, 0.40, 1, 4.582, 4.588, 0.017),
    TableCase(42, 0.40, 2, 6.248, 6.243, 0.020),
    TableCase(44, 0.20, 1, 1.110, 1.118, 0.007),
    TableCase(44, 0.20, 2, 1.690, 1.675, 0.009),
    TableCase(44, 0.40, 1, 3.948, 3.957, 0.017),
    TableCase(44, 0.40, 2, 5.647, 5.622, 0.021),
)
