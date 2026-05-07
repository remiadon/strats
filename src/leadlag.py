"""
Lead-lag portfolios via path signatures (Levy area).

Background
----------
The Levy area between two paths X and Y is the antisymmetric part of the
level-2 signature ::

    L(X, Y) = (1/2) (S^{XY} - S^{YX})

It is exactly the [X, Y] bracket coordinate of the level-2 log-signature
(shuffle identity: ``S^X * S^Y = S^{XY} + S^{YX}``).

We keep the Levy areas in **upper-triangular** form only - one expression
per ordered pair ``(ti, tj)`` with ``ti < tj`` in channel order - and read
the sign of ``L_{ij}`` to route trades:

    L_{ij} > 0  ⇒  ``ti`` led ``tj`` over the window  ⇒  ``tj`` is follower
    L_{ij} < 0  ⇒  ``tj`` led ``ti``                  ⇒  ``ti`` is follower

The follower's position is sized by the leader's cumulative log-return
over the same window (level-1 of the rolling signature) - so a follower
only earns exposure when its leader has actually ticked.

Implementation
--------------
* No custom classes - functions only.
* Wide format throughout: one column per ticker, one row per date.
* Heavy expressions are split into staged DataFrames (Levy areas + level-1
  → pair contributions → weights) so the polars expression DAG stays
  small enough for IPython kernels to handle big baskets.
* Data prep (download, pivot, ffill) is intentionally NOT in this module
  - use ``src.io.get_stocks`` and a one-line ``pivot`` in the notebook.
"""

from itertools import combinations
from typing import Iterable, List, Mapping, Tuple

import polars as pl
import polars.selectors as cs

from . import signatures as sg


# ---------------------------------------------------------------------------
# 1. Building blocks
# ---------------------------------------------------------------------------

def log_return_exprs(tickers: Iterable[str]) -> List[pl.Expr]:
    """One polars expression per ticker for log-return increments."""
    return [pl.col(t).log().diff().fill_null(0.0).alias(t) for t in tickers]


def levy_area_exprs(
    tickers: Tuple[str, ...],
    window: int,
) -> Mapping[Tuple[str, str], pl.Expr]:
    """
    Upper-triangular pairwise rolling Levy areas - one expression per
    ``(ti, tj)`` with ``ti < tj`` in channel order.

    Built from `signatures.py`::

        log( rolling( signature(*log_returns, level=2), window ) )

    The bracket entries (length-2 keys) of the resulting Hall-basis
    coordinate dict are exactly the Levy areas.
    """
    sigs = sg.signature(*log_return_exprs(tickers), level=2)
    log_sigs = sg.log(sg.rolling(sigs, window_size=window))
    return {k: v for k, v in log_sigs.items() if len(k) == 2}


def rolling_log_return_exprs(
    tickers: Tuple[str, ...],
    window: int,
) -> Mapping[str, pl.Expr]:
    """
    Cumulative log-return of each ticker over the rolling window - the
    level-1 entry of the same rolling signature, reused as the leader's
    "tick" magnitude in `signal`.
    """
    sigs = sg.signature(*log_return_exprs(tickers), level=2)
    rolling_sigs = sg.rolling(sigs, window_size=window)
    return {k[0]: v for k, v in rolling_sigs.items() if len(k) == 1}


# ---------------------------------------------------------------------------
# 2. Pair-trading signal
# ---------------------------------------------------------------------------

def signal(df_wide: pl.DataFrame, window: int = 60) -> pl.DataFrame:
    """
    Daily leader-follower pair-trading weights from a wide price frame.

    For each upper-triangular pair ``(ti, tj)`` with ``ti < tj``::

        if L_{ij} > 0:  ti leads, follower=tj  →  w_{tj} += L_{ij}  · S_{ti}
        if L_{ij} < 0:  tj leads, follower=ti  →  w_{ti} += -L_{ij} · S_{tj}

    where ``S_{t}`` is the cumulative log-return of the leader over the
    same rolling window.  Sum across pairs, then L1-normalise so
    ``Σ |w_i| = 1``.  No ``+/-`` symmetric matrix - we only ever look at
    each pair once via ``itertools.combinations``.

    Computed in three staged DataFrames so the polars expression DAG
    stays manageable.
    """
    tickers = tuple(c for c in df_wide.columns if c != "date")

    # Stage 1 — materialise rolling Levy areas (level 2) and rolling
    # cumulative log-returns (level 1) in one wide intermediate frame.
    levys = levy_area_exprs(tickers, window)
    rets = rolling_log_return_exprs(tickers, window)
    df_int = df_wide.select(
        "date",
        **{f"__lv__{a}__{b}": v for (a, b), v in levys.items()},
        **{f"__r__{t}": v for t, v in rets.items()},
    )

    # Stage 2 — per-ticker follower position = sum over pairs where the
    # other end of the pair is the leader (positive contribution only).
    weights = {t: pl.lit(0.0) for t in tickers}
    for ti, tj in combinations(tickers, 2):
        lv = pl.col(f"__lv__{ti}__{tj}")
        r_i = pl.col(f"__r__{ti}")
        r_j = pl.col(f"__r__{tj}")
        # ti leads tj  →  contribute to tj
        weights[tj] = weights[tj] + lv.clip(lower_bound=0) * r_i
        # tj leads ti  →  contribute to ti
        weights[ti] = weights[ti] + (-lv).clip(lower_bound=0) * r_j
    df_w = df_int.select("date", **weights)

    # Stage 3 — L1-normalise to unit gross exposure, pure selector
    # arithmetic so we never enumerate `tickers` again.
    return (
        df_w
        .with_columns(cs.exclude("date") / pl.sum_horizontal(cs.exclude("date").abs()))
        .fill_nan(0.0)
        .fill_null(0.0)
    )


# ---------------------------------------------------------------------------
# 3. Backtest
# ---------------------------------------------------------------------------

def _resample_weights(weights: pl.DataFrame, rebal_every: str) -> pl.DataFrame:
    """
    Snapshot end-of-period weights, carry forward via asof-join, then
    shift by one row so weights set at close of day t earn day t+1's
    return - no look-ahead.
    """
    rebal = (
        weights.sort("date")
        .group_by_dynamic("date", every=rebal_every, closed="left")
        .agg(
            pl.col("date").last().alias("rebal_date"),
            cs.exclude("date").last(),
        )
        .drop("date")
        .rename({"rebal_date": "date"})
        .sort("date")
    )
    return (
        weights.select("date").sort("date")
        .join_asof(rebal, on="date", strategy="backward")
        .with_columns(cs.exclude("date").shift(1))
    )


def backtest(
    df_wide: pl.DataFrame,
    weights: pl.DataFrame,
    rebal_every: str = "1mo",
) -> pl.DataFrame:
    """
    Apply daily-shift / monthly-rebal to ``weights`` against the daily
    simple-returns of ``df_wide``.  Returns ``(date, pnl, equity)``.
    """
    daily_w = _resample_weights(weights, rebal_every)
    cols = [c for c in daily_w.columns if c != "date"]
    rets = df_wide.with_columns(cs.exclude("date").pct_change().fill_null(0.0))

    return (
        daily_w
        .join(rets.rename({c: f"__r__{c}" for c in cols}), on="date")
        .select(
            "date",
            pnl=pl.sum_horizontal(
                pl.col(c).fill_null(0.0) * pl.col(f"__r__{c}") for c in cols
            ),
        )
        .with_columns(equity=(pl.col("pnl") + 1.0).cum_prod())
    )


def buy_and_hold(df_wide: pl.DataFrame) -> pl.DataFrame:
    """
    Equally-weighted buy-and-hold baseline: invest 1/N at t=0, never
    rebalance.  Returns ``(date, pnl, equity)``.
    """
    n = df_wide.width - 1
    return (
        df_wide.select(
            "date",
            equity=pl.sum_horizontal(cs.exclude("date") / cs.exclude("date").first())
                   / pl.lit(n),
        )
        .with_columns(pnl=pl.col("equity").pct_change().fill_null(0.0))
        .select("date", "pnl", "equity")
    )


# ---------------------------------------------------------------------------
# 4. Reporting helpers
# ---------------------------------------------------------------------------

def monthly_returns(pnl: pl.DataFrame) -> pl.DataFrame:
    """Compound a daily-pnl frame into monthly returns."""
    return (
        pnl.group_by_dynamic("date", every="1mo")
        .agg(((pl.col("pnl") + 1.0).product() - 1.0).alias("monthly_return"))
        .sort("date")
    )


def perf_summary(pnl: pl.DataFrame, periods_per_year: int = 365) -> dict:
    """
    Headline stats - all expressed in pure polars.  ``periods_per_year``
    is 365 for crypto (24/7), 252 for equity.
    """
    sqrt_ppy = pl.lit(periods_per_year ** 0.5)
    equity = (pl.col("pnl") + 1.0).cum_prod()
    return pnl.select(
        cagr=equity.last() ** (pl.lit(periods_per_year) / pl.len()) - 1.0,
        vol=pl.col("pnl").std() * sqrt_ppy,
        sharpe=(pl.col("pnl").mean() / pl.col("pnl").std()) * sqrt_ppy,
        max_dd=(equity / equity.cum_max() - 1.0).min(),
    ).row(0, named=True)


def plot_monthly_returns(
    pnl: pl.DataFrame,
    title: str = "Monthly returns",
    ax=None,
):
    """
    Heatmap of monthly returns - rows=year, columns=month.  Equivalent
    to ``leadlag_port.plot_monthly_returns`` in the original repo.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    m = monthly_returns(pnl).with_columns(
        year=pl.col("date").dt.year(),
        month=pl.col("date").dt.month(),
    )
    pivoted = (
        m.pivot(values="monthly_return", index="year", on="month").sort("year")
    )
    years = pivoted.get_column("year").to_list()
    month_cols = sorted(
        (c for c in pivoted.columns if c != "year"),
        key=lambda c: int(c),
    )
    matrix = pivoted.select(month_cols).to_numpy() * 100.0  # %

    if ax is None:
        _, ax = plt.subplots(figsize=(11, max(2.5, 0.55 * len(years) + 1)))
    cmap = plt.get_cmap("RdYlGn")
    vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(len(month_cols)))
    ax.set_xticklabels([month_names[int(c) - 1] for c in month_cols])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j, i, f"{v:+.1f}%",
                ha="center", va="center", fontsize=8,
                color="black" if abs(v) < 0.6 * vmax else "white",
            )
    plt.colorbar(im, ax=ax, label="Return (%)", fraction=0.025, pad=0.02)
    return ax


def plot_equity_curves(
    curves: Mapping[str, pl.DataFrame],
    title: str = "Equity curves",
    ax=None,
):
    """Overlay equity curves from a dict ``{label: pnl_frame}``."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    for name, pnl in curves.items():
        ax.plot(pnl.get_column("date"), pnl.get_column("equity"), label=name)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    return ax


def plot_levy_triangle(
    df_wide: pl.DataFrame,
    window: int,
    title: str | None = None,
    ax=None,
):
    """
    Heatmap of the most-recent rolling Levy-area matrix, drawn as a
    strict upper-triangle (no symmetric mirror, no ``+/-`` duplication).
    Cell ``(i, j)`` reads "did ``ti`` lead ``tj`` over the last
    ``window`` days" - red = yes, blue = no.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    tickers = tuple(c for c in df_wide.columns if c != "date")
    levys = levy_area_exprs(tickers, window)
    last = df_wide.select(
        *[v.tail(1).alias(f"{a}|{b}") for (a, b), v in levys.items()]
    ).row(0, named=True)

    n = len(tickers)
    M = np.full((n, n), np.nan)
    ix = {t: i for i, t in enumerate(tickers)}
    for k, v in last.items():
        a, b = k.split("|")
        M[ix[a], ix[b]] = v  # only the upper-triangular cell

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = float(np.nanmax(np.abs(M))) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(tickers, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(tickers, fontsize=8)
    ax.set_title(title or f"Pairwise Levy area, last {window}d  (row leads col, upper triangle only)")
    plt.colorbar(im, ax=ax, fraction=0.04)
    return ax
