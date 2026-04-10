import polars as pl
import polars.selectors as cs
from operator import __or__
from itertools import product
from math import factorial
from typing import Mapping, Tuple, Union, List

import polars as pl
import polars.selectors as cs

from extract.io import dump, get_sources, get_sources_config


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


def named_signature(df: Union[pl.DataFrame, pl.LazyFrame], **kwargs):
    cols = cs.expand_selector(df, cs.numeric())
    exprs = signature(*map(pl.col, cols), **kwargs)  # test with level=3
    return {f'S({",".join(k)})': v for k, v in exprs.items()}

# TODO : try the transform API from functime
#@memory.cache
def compute_signatures( 
        sources: List[str],
        target: str,
        level: int = 3, 
        group_by = None,
        index_column='date', 
        periods: Tuple[int] = (7, 30, 90),
    ):
    """
    signatures are supposed to build out non-linear signal so we caching/discarding them based on a linear Pearson correlation should be enough
    """
    df = get_sources(sources).lazy()
    index = (group_by or [], index_column)
    df = df.drop_nulls(subset=list(index)).sort(index)

    cols = cs.expand_selector(df, cs.numeric())
    sigs = signature(*(pl.col(col).cast(pl.Float64).pct_change() for col in cols), level=level)  # test with level=3
    feats = dict()
    for w in periods:
        feats.update({f'S({",".join(k)})_{w}d': v.over('ticker') for k, v in rolling(sigs, window_size=w).items()})

    feats = {k: v for k, v in feats.items() if target in k}
    print(f"Filtered signatures to those containing the target '{target}', starting from {len(sigs)}, resulting in {len(feats)} features.")
    result, profile =  df.select(*index, **feats).profile(engine='streaming')
    # TODO : remove co-linear relations and keep lower-level nodes in case of conflicts
    print(profile.select('node', pl.col.end.sub(pl.col.start).alias('tot_time')))
    return result


if __name__ == '__main__':
    import polars as pl
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sources", help="see params.yaml", type=str, nargs='+', required=True)
    parser.add_argument("-t", "--target", help="filter signatures to those containing this target feature", type=str, required=True)
    parser.add_argument("-o", "--output", help="output file", type=str, required=True)
    parser.add_argument("-g", "--group-by", help="group key", type=str, default=None)
    parser.add_argument("-p", "--periods", type=int, default=30, nargs='+')
    kw = parser.parse_args()
    output = kw.__dict__.pop('output')
    sigs = compute_signatures(**kw.__dict__, index_column='date')
    dump(sigs, output_path=output, key='date' if kw.group_by is None else [kw.group_by, 'date'])
