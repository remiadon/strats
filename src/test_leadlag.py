"""
Sanity tests for `src/leadlag.py`.

Run from the project root with the project venv active::

    python -m src.test_leadlag

Checks
------
1. The Hall-basis bracket from `log(signature(A.diff(), B.diff(), level=2))`
   matches the analytic Levy area on synthetic piecewise-linear paths.
2. The rolling version matches a brute-force python loop over windows.
3. End-to-end ``signal -> backtest`` produces a finite, well-behaved
   equity curve on synthetic data and ``buy_and_hold`` likewise.
4. The pair-trading weights satisfy the unit-gross invariant
   ``Σ |w| ∈ {0, 1}``.
"""

import datetime as dt
import numpy as np
import polars as pl

from . import signatures as sg
from . import leadlag as ll


def _analytic_levy(A: np.ndarray, B: np.ndarray) -> float:
    """(1/2) ∫ ((A-A0) dB - (B-B0) dA) for a piecewise-linear path."""
    A0, B0 = A[0], B[0]
    n = len(A)
    return 0.5 * sum(
        (A[k] - A0) * (B[k + 1] - B[k]) - (B[k] - B0) * (A[k + 1] - A[k])
        for k in range(n - 1)
    )


def test_global_levy() -> None:
    np.random.seed(0)
    n = 50
    A = np.cumsum(np.random.randn(n))
    B = np.cumsum(np.random.randn(n))
    df = pl.DataFrame({"A": A, "B": B})

    sigs = sg.signature(
        pl.col("A").diff().fill_null(0.0),
        pl.col("B").diff().fill_null(0.0),
        level=2,
    )
    levy_lib = df.select(sg.log(sigs)[("A", "B")].tail(1)).item()
    levy_ref = _analytic_levy(A, B)
    assert np.isclose(levy_lib, levy_ref), (levy_lib, levy_ref)
    print(f"global Levy area : lib={levy_lib:+.6f}  analytic={levy_ref:+.6f}  OK")


def test_rolling_levy() -> None:
    np.random.seed(1)
    n, w = 200, 30
    A = np.cumsum(np.random.randn(n))
    B = np.cumsum(np.random.randn(n))
    df = pl.DataFrame({"A": A, "B": B})

    sigs = sg.signature(
        pl.col("A").diff().fill_null(0.0),
        pl.col("B").diff().fill_null(0.0),
        level=2,
    )
    rolling_lib = (
        df.select(sg.log(sg.rolling(sigs, window_size=w))[("A", "B")])
        .to_numpy()
        .ravel()
    )
    brute = np.full(n, np.nan)
    for t in range(w - 1, n):
        brute[t] = _analytic_levy(A[t - w + 1 : t + 1], B[t - w + 1 : t + 1])
    mask = ~np.isnan(brute)
    err = np.max(np.abs(rolling_lib[mask] - brute[mask]))
    assert err < 1e-9, err
    print(f"rolling Levy ({mask.sum()} windows): max abs err = {err:.2e}  OK")


def test_signal_invariants_and_e2e() -> None:
    np.random.seed(7)
    n, K = 400, 6
    dates = pl.date_range(
        start=dt.date(2022, 1, 1),
        end=dt.date(2022, 1, 1) + dt.timedelta(days=n - 1),
        eager=True,
    )
    rets = np.random.randn(n, K) * 0.02
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    wide = pl.DataFrame({"date": dates, **{f"c{i}": prices[:, i] for i in range(K)}})

    weights = ll.signal(wide, window=30)
    gross = weights.select(
        gross_w=pl.sum_horizontal(pl.exclude("date").abs())
    ).get_column("gross_w").to_numpy()
    # Either 0 (degenerate burn-in) or ~1 (post burn-in, normalised gross)
    bad = ~((np.abs(gross) < 1e-10) | (np.abs(gross - 1.0) < 1e-10))
    assert not bad.any(), gross[bad][:5]
    print(f"signal invariants: Σ|w| ∈ {{0, 1}} on every row  OK")

    pnl_ll = ll.backtest(wide, weights, rebal_every="1mo")
    pnl_bh = ll.buy_and_hold(wide)
    assert pnl_ll.height == wide.height
    assert pnl_bh.height == wide.height
    assert pnl_ll["equity"].is_finite().all()
    assert pnl_bh["equity"].is_finite().all()
    print(f"backtest equity OK : pair-trading final={pnl_ll['equity'][-1]:.4f}  "
          f"B&H final={pnl_bh['equity'][-1]:.4f}")

    monthly = ll.monthly_returns(pnl_ll)
    assert monthly.height >= 1
    print(f"monthly_returns rows: {monthly.height}  OK")


def test_pair_routing() -> None:
    """
    Build two assets where A clearly leads B (B's return at t = A's at
    t-1).  We expect pair-trading to hold a *long* position in B (the
    follower) when A has just gone up - the position must move with the
    sign of A's recent return.
    """
    np.random.seed(11)
    n = 200
    rA = np.random.randn(n) * 0.01
    rB = np.zeros(n)
    rB[1:] = rA[:-1]            # B perfectly follows A by one day
    pA = 100 * np.exp(np.cumsum(rA))
    pB = 100 * np.exp(np.cumsum(rB))
    dates = pl.date_range(
        dt.date(2022, 1, 1),
        dt.date(2022, 1, 1) + dt.timedelta(days=n - 1),
        eager=True,
    )
    wide = pl.DataFrame({"date": dates, "A": pA, "B": pB})

    weights = ll.signal(wide, window=30).tail(1)
    wA = weights["A"].item()
    wB = weights["B"].item()
    # A leads → follower B should carry (almost) all of the gross weight,
    # and A should carry essentially zero.
    assert abs(wB) > 0.9, (wA, wB)
    assert abs(wA) < 0.1, (wA, wB)
    # Sign of B's weight should mirror A's most recent cumulative return
    # over the window — i.e. positive iff A trended up.
    sign_A_recent = np.sign(np.log(pA[-1] / pA[-31]))
    assert np.sign(wB) == sign_A_recent, (wB, sign_A_recent)
    print(f"pair routing OK   : leader=A → wA={wA:+.3f}, wB={wB:+.3f} "
          f"(sign matches recent A trend)")


if __name__ == "__main__":
    test_global_levy()
    test_rolling_levy()
    test_signal_invariants_and_e2e()
    test_pair_routing()
    print("\nALL LEAD-LAG TESTS PASSED")
