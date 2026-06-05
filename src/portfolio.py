"""
Portfolio evaluation.

``evaluate`` is the single entry-point for deriving risk/return metrics from
a weight mapping and a price history.  Weights are a plain dict[str, float]
summing to 1 — no opinion on how they were derived.

Intended MCP dependency graph:

    get_stocks()
        ├── build_portfolio(...) -> dict[str, float]   # construction tool
        └── evaluate(weights, prices) -> dict          # evaluation tool
                     ↑
             weights come from build_portfolio OR a portfolio registry
"""
import numpy as np
import polars as pl


def evaluate(
    weights: dict[str, float],
    prices: pl.DataFrame,
    *,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute risk/return metrics for a set of weights over a price history.

    Parameters
    ----------
    weights:
        Ticker -> weight mapping. Weights must sum to 1. Tickers absent from
        *prices* are dropped and the remainder is renormalised, so the function
        degrades gracefully when the universe shifts between construction and
        evaluation.
    prices:
        Wide DataFrame with one column per ticker. A ``date`` column is ignored
        if present. Missing values are forward- then backward-filled.
    risk_free_rate:
        Annualised risk-free rate for the Sharpe ratio (default 0).

    Returns
    -------
    dict with keys: annual_return, annual_vol, sharpe, max_drawdown
    """
    available = {t: w for t, w in weights.items() if t in prices.columns}
    if not available:
        raise ValueError("None of the portfolio tickers are present in the price DataFrame.")

    tickers = list(available)
    total_w = sum(available.values())
    w = np.array([available[t] / total_w for t in tickers])

    price_matrix = (
        prices
        .select([pl.col(t).cast(pl.Float64) for t in tickers])
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .to_numpy()
    )  # (T, M)

    # Buy-and-hold value series, normalised to 1 at inception
    portfolio_value = price_matrix @ w / float(price_matrix[0] @ w)
    daily_returns   = portfolio_value[1:] / portfolio_value[:-1] - 1

    annual_return = float((1 + daily_returns.mean()) ** 252 - 1)
    annual_vol    = float(daily_returns.std() * np.sqrt(252))
    sharpe        = (annual_return - risk_free_rate) / annual_vol if annual_vol else float("nan")
    running_peak  = np.maximum.accumulate(portfolio_value)
    max_drawdown  = float(((running_peak - portfolio_value) / running_peak).max())

    return {
        "annual_return": annual_return,
        "annual_vol":    annual_vol,
        "sharpe":        sharpe,
        "max_drawdown":  max_drawdown,
    }
