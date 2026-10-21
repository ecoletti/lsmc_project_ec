"""Black-Scholes helpers for the European component of the benchmark."""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def european_put_price(s0: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    """Return the Black-Scholes price of a European put without dividends."""
    if maturity <= 0:
        return max(strike - s0, 0.0)
    if sigma <= 0:
        forward = s0 * exp(rate * maturity)
        return exp(-rate * maturity) * max(strike - forward, 0.0)

    vol_sqrt_t = sigma * sqrt(maturity)
    d1 = (log(s0 / strike) + (rate + 0.5 * sigma * sigma) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return strike * exp(-rate * maturity) * normal_cdf(-d2) - s0 * normal_cdf(-d1)


def american_put_price_barone_adesi_whaley(
    s0: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    dividend_yield: float = 0.0,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Approximate an American put using Barone-Adesi and Whaley (1987)."""
    if maturity <= 0:
        return max(strike - s0, 0.0)
    if sigma <= 0:
        forward = s0 * exp((rate - dividend_yield) * maturity)
        return max(strike - s0, exp(-rate * maturity) * max(strike - forward, 0.0))

    european_price = european_put_price(s0, strike, rate, sigma, maturity)
    if rate <= 0.0 and dividend_yield <= 0.0:
        return european_price

    critical_spot = _critical_spot_for_american_put(
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        dividend_yield=dividend_yield,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if s0 <= critical_spot:
        return strike - s0

    cost_of_carry = rate - dividend_yield
    vol_squared = sigma * sigma
    n = 2.0 * cost_of_carry / vol_squared
    m = 2.0 * rate / vol_squared
    k_factor = 1.0 - exp(-rate * maturity)
    q1 = (-(n - 1.0) - sqrt((n - 1.0) * (n - 1.0) + 4.0 * m / k_factor)) / 2.0
    d1 = (log(critical_spot / strike) + (cost_of_carry + 0.5 * vol_squared) * maturity) / (sigma * sqrt(maturity))
    early_exercise_premium = -(critical_spot / q1) * (
        1.0 - exp((cost_of_carry - rate) * maturity) * normal_cdf(-d1)
    )
    return european_price + early_exercise_premium * (s0 / critical_spot) ** q1


def _critical_spot_for_american_put(
    *,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    dividend_yield: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    cost_of_carry = rate - dividend_yield
    vol_squared = sigma * sigma
    sqrt_t = sqrt(maturity)
    n = 2.0 * cost_of_carry / vol_squared
    m = 2.0 * rate / vol_squared
    k_factor = 1.0 - exp(-rate * maturity)
    if k_factor <= 0.0:
        return strike

    q1 = (-(n - 1.0) - sqrt((n - 1.0) * (n - 1.0) + 4.0 * m / k_factor)) / 2.0
    q1_perpetual = (-(n - 1.0) - sqrt((n - 1.0) * (n - 1.0) + 4.0 * m)) / 2.0
    perpetual_exercise_boundary = strike / (1.0 - 1.0 / q1_perpetual)
    h1 = (cost_of_carry * maturity - 2.0 * sigma * sqrt_t) * strike / (strike - perpetual_exercise_boundary)
    spot = perpetual_exercise_boundary + (strike - perpetual_exercise_boundary) * exp(h1)

    for _ in range(max_iterations):
        d1 = (log(spot / strike) + (cost_of_carry + 0.5 * vol_squared) * maturity) / (sigma * sqrt_t)
        european_price = european_put_price(spot, strike, rate, sigma, maturity)
        lhs = strike - spot
        rhs = european_price - (spot / q1) * (
            1.0 - exp((cost_of_carry - rate) * maturity) * normal_cdf(-d1)
        )
        slope = (
            -exp((cost_of_carry - rate) * maturity) * normal_cdf(-d1) * (1.0 - 1.0 / q1)
            - (
                1.0
                + exp((cost_of_carry - rate) * maturity) * normal_pdf(d1) / (sigma * sqrt_t)
            )
            / q1
        )
        next_spot = (strike - rhs + slope * spot) / (1.0 + slope)
        if abs(next_spot - spot) <= tolerance * strike:
            return next_spot
        spot = next_spot

    return spot
