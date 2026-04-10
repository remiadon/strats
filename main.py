import datetime as dt
from itertools import product
from math import factorial
from typing import Mapping, Tuple

import numpy as np
import polars as pl
import polars.selectors as cs


def signature(*exprs: pl.Expr, level: int = 2) -> Mapping[Tuple[str], pl.Expr]:
    """
    Computes path signature components up to any arbitrary level.
    Uses only Polars Expressions and aligns with the piecewise linear assumption.

    Return the global cumulative signature as a mapping of word tuples to
    Polars expressions

    In practice every element of `exprs` should express increments : 
    it's left to the end-user to choose how those increments are computed. This usually involves calling
    `.diff()` or `.pct_change()` or a log/return transform, depending on the use case at hand.

    Parameters
    ----------
    *exprs : pl.Expr    - Expressions. In practice this is advised to pass increments instead of raw fetures
    level  : int       — signature truncation level

    Returns
    -------
    Mapping[Tuple[str, ...], pl.Expr]
        Keys are word tuples, e.g. ``("A",)``, ``("A", "B")``.
        Values are unevaluated Polars expressions.
    """
    assert not any((expr.meta.has_multiple_outputs() for expr in exprs)), "all expressions in `exprs` must project to a single expression"
    get_name = lambda e: e.meta.output_name()

    # 1. Prepare Base Increments (dX)
    incs = {get_name(e): e for e in exprs}
    # 2. Key Mapping (sigs): 
    # Keys are tuples of indices: (0, 1) 
    # Values are Polars Expressions
    sigs = dict()

    # --- Level 1 ---
    sigs = {(k,): v.cum_sum() for k, v in incs.items()}

    # --- Higher Levels (2 to level) ---
    for d in range(2, level + 1):
        for word in product(map(get_name, exprs), repeat=d):
            # The "Local Integral" of the segment itself: (1/d!) * dX_1 * dX_2 * ... * dX_d
            local_integral = pl.lit(1 / factorial(d))
            for idx in word:
                local_integral = local_integral * incs[idx]

            # The "Recursive Update": S_new = S_old + (Iterative Prefix Sums) + local_integral
            # We follow the formula: Delta S_w = Sum_{split} [ S_prefix_old * Delta S_suffix ]
            term = local_integral
            
            for split in range(1, d):
                prefix_word = word[:split]
                suffix_word = word[split:]
                
                # S_old of the prefix (value before this current jump)
                s_prefix_prev = sigs[prefix_word].shift(1).fill_null(0)
                
                # Delta S of the suffix (the local integral of the remaining indices)
                delta_s_suffix = pl.lit(1 / factorial(len(suffix_word)))
                for idx in suffix_word:
                    delta_s_suffix = delta_s_suffix * incs[idx]

                term = term + (s_prefix_prev * delta_s_suffix)

            sigs[word] = term.cum_sum()

    return sigs

def rolling(
    sigs: Mapping[Tuple[str, ...], pl.Expr],
    window_size: int,
) -> Mapping[Tuple[str, ...], pl.Expr]:
    """
    Build rolling signature expressions

    No DataFrame is touched here.  The returned mapping is applied to a
    DataFrame by the caller, e.g.::

        sigs = global_signature(pl.col("A").diff(), pl.col("B").diff(), level=2)
        gs_df = df.select([v.alias(",".join(k)) for k, v in sigs.items()])
        result_30 = gs_df.select([v.alias(",".join(k)) for k, v in rolling(sigs, 30).items()])

    Parameters
    ----------
    sigs         : Mapping[Tuple[str, ...], pl.Expr]
        Output of ``global_signature()``.
    window_size : int

    Returns
    -------
    Mapping[Tuple[str, ...], pl.Expr]
        key: word tuple. 
        value: a Polars expression implementing Chen's lag-subtraction for that word
        and window — ready to be passed to ``.select()`` on the materialised
        global signature DataFrame.

    Implementation note — Chen's identity
    --------------------------------------
    The local helper ``chen(word, lag)`` recovers the window-local signature
    of ``word`` from the global cumulative signature columns via:

        S^word[s..e] = (CS^word[e] − CS^word[s])
                     − Σ_{splits} CS^prefix[s] · S^suffix[s..e]

    where ``CS^word[s] = pl.col(word).shift(lag).fill_null(0)``.
    The recursion bottoms out at level-1 words (no splits), which reduce to
    a plain lag-subtraction.  Every sub-expression is O(1) column arithmetic.
    """
    def chen(word: tuple[str, ...], lag: int) -> pl.Expr:
        delta = sigs[word] - sigs[word].shift(lag).fill_null(0.0)
        correction = pl.lit(0.0)
        for split in range(1, len(word)):
            prefix, suffix = word[:split], word[split:]
            correction = correction + (
                sigs[prefix].shift(lag).fill_null(0.0) * chen(suffix, lag)
            )
        return delta - correction

    return {word: chen(word, window_size - 1) for word in sigs}


    
if __name__ == '__main__':
    ### IMPORTS
    import esig
    import iisignature
    import numpy.testing
    import polars.testing
    from sktime.transformations.panel.signature_based import SignatureTransformer
    
    
    ### TESTS ON EXPRESSIONS / MAPPINGS ONLY
    to_diff = lambda s: pl.col(s).diff().fill_null(0.0)
    sigs = signature(*map(to_diff, 'ABC'), level=2)
    assert len(signature(*map(to_diff, 'ABC'), level=3)) == 39
    assert len(sigs) == 12
    print(sigs)
    
    ### TESTS ON A PLAIN DF
    feats = {",".join(k): v.tail(1) for k, v in sigs.items()}
    df = pl.DataFrame({
        "time": pl.date_range(start=dt.date(2025, 1, 1), end=dt.date(2025, 1, 4), eager=True),
        "A": [0, 1, 1, 0],
        "B": [0, 0, 1, 1],
        "C": [1, 2, 3, 2]
    })

    # Simply assess we get the same values from these 2 baseline libraries
    numpy.testing.assert_array_equal(
        iisignature.sig(df.drop('time').to_numpy(), 2),
        esig.stream2sig(df.drop('time').to_numpy().astype(float), depth=2)[1:],
    )

    ### now test a simple signature computation gives us what we want
    numpy.testing.assert_array_equal(
        iisignature.sig(df.drop('time').to_numpy(), 2),
        df.drop('time').select(**feats).row(0)
    )
    print(esig.stream2logsig(df.drop('time').to_numpy().astype(float), depth=2))
    
    
    ### TESTS ON A SPARSER DF : usually the result of a left join between datasets of different cadence
    df = pl.DataFrame({
        "time": pl.date_range(start=dt.date(2025, 1, 1), end=dt.date(2025, 1, 4), eager=True),
        "A": [0, 1, 1, 0],
        "B": [0, None, 1, None], # let us say be is observed twice less frequently than A
        "C": [1, 2, 3, 2]
    })

    #numpy.testing.assert_array_equal(. # FIXME esig/iisignature returns nans
    #    np.nan_to_num(esig.stream2sig(df.drop('time').to_numpy().astype(float), depth=2)[1:]), # This contains nans, let's fill them
    #    df.drop('time').select(**feats).row(0)
    #)

    #### TEST ROLLING SIGNATURE
    rolling_baseline = pl.DataFrame([{
        'time': datetime, 
        'sig': np.nan_to_num(esig.stream2sig(_df.drop('time').to_numpy().astype(float), depth=2)[1:])
        } for datetime, _df in df.rolling(index_column='time', period='2d')
    ]).select(pl.col.sig.list.to_array(12)).get_column('sig').to_numpy()

    numpy.testing.assert_array_equal(
        rolling_baseline,
        df.rolling(index_column='time', period='2d').agg(**feats).select(cs.list().list.get(0)).to_numpy()
    )
    print(df.rolling(index_column='time', period='2d').agg(**feats).select('time', cs.list().list.get(0)))

    window_size = (2, 10, 30)
    for num_col in (5, 15):
        for size in (10_000, 50_000):
            df = pl.from_numpy(np.random.rand(size, num_col)).lazy()
            sigs = signature(*map(to_diff, df.columns), level=2)

            rolling_baselines = [
                df.rolling(
                    index_column=pl.int_range(0, pl.len()), period=f'{w}i'
                ).agg(**{",".join(k) + f"_{w}": v.tail(1) for k, v in sigs.items()})
                for w in window_size
            ]
            rolling_baseline = pl.concat(rolling_baselines, how='align')

            fast_rolling = df.select(**{",".join(k) + f"_{w}": v for w in window_size for k, v in rolling(sigs, window_size=w).items()})

            rolling_baseline, rolling_profile = rolling_baseline.profile(engine='streaming')
            fast_rolling, fast_profile = fast_rolling.profile(engine='streaming')

            rolling_profile.filter(pl.col.node.str.contains('group_by_rollin|optimization')) # make sure we don't account for concat/join -> fair benchmark
            tot_time = pl.col.end.sub(pl.col.start).sum()
            print(f"""
                    FAST ROLLING SPEEDUP for shape = ({size},{num_col}) and window_size {window_size}: 
                    {(rolling_profile.select(tot_time) / fast_profile.select(tot_time)).item(0, 0)}
            """)
            polars.testing.assert_frame_equal(rolling_baseline.select(cs.list().list.get(0)), fast_rolling)

    
    ### TEST CONSISTENT WITH SKTIME
    signature_transform = SignatureTransformer(
        augmentation_list=("basepoint",),
        window_name="global",
        window_depth=None,
        window_length=None,
        window_step=None,
        rescaling=None,
        sig_tfm="signature",
        depth=2,
    )

    df = pl.DataFrame({ # same df as before, but insert zeros at the very first position = "basepoint" transform
        "A": [0, 0, 1, 1, 0],
        "B": [0, 0, 0, 1, 1],
        "C": [0, 1, 2, 3, 2]
    })

    numpy.testing.assert_array_equal(
        signature_transform.fit_transform(df.to_numpy()),
        df.select(**feats).to_numpy()
    )
