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

def log(
    sigs: Mapping[Tuple[str, ...], pl.Expr],
) -> Mapping[Tuple[str, ...], pl.Expr]:
    """
    Compute the log-signature from a pre-computed signature mapping.

    Applies ``log(1 + S) = S − S²/2 + S³/3 − …`` in the truncated tensor
    algebra, then projects onto the **Hall basis** (matching ``esig`` /
    ``iisignature`` / ``sktime``).

    The truncation level is inferred from the longest key in *sigs*.

    Accepts any ``Mapping[Tuple[str, ...], pl.Expr]``, so composes freely::

        log(signature(...))           # global log-signature
        log(rolling(signature(...)))  # rolling log-signature
    """
    level = max(len(k) for k in sigs)
    channels = [k[0] for k in sigs if len(k) == 1]

    # ── Step 1: tensor-algebra logarithm ─────────────────────────────
    # log(1+S) = S − S⊗S/2 + S⊗S⊗S/3 − …   (truncated at `level`)
    tensor_log: dict[Tuple[str, ...], pl.Expr] = dict(sigs)
    s_power: dict[Tuple[str, ...], pl.Expr] = dict(sigs)

    for k in range(2, level + 1):
        new_power: dict[Tuple[str, ...], pl.Expr] = {}
        for w1, e1 in s_power.items():
            for w2, e2 in sigs.items():
                w_cat = w1 + w2
                if len(w_cat) > level:
                    continue
                new_power[w_cat] = new_power.get(w_cat, pl.lit(0.0)) + e1 * e2
        s_power = new_power
        coeff = ((-1) ** (k + 1)) / k
        for word, expr in s_power.items():
            tensor_log[word] = tensor_log.get(word, pl.lit(0.0)) + pl.lit(coeff) * expr

    # ── Step 2: build Hall basis & expand each element to tensor words ──
    # A Hall element is either a letter (str) or a bracket (h_left, h_right).
    # We represent each as a tree and also cache its tensor-word expansion
    # (dict of {word_tuple: int coefficient}) and its "foliage" (the leading
    # tensor word = concatenation of its leaves left-to-right).

    def _foliage(h) -> Tuple[str, ...]:
        """Concatenation of the leaves of a Hall tree, left to right."""
        if isinstance(h, str):
            return (h,)
        return _foliage(h[0]) + _foliage(h[1])

    def _expand(h) -> dict[Tuple[str, ...], int]:
        """Expand a Hall bracket tree into the tensor algebra."""
        if isinstance(h, str):
            return {(h,): 1}
        e1, e2 = _expand(h[0]), _expand(h[1])
        out: dict[Tuple[str, ...], int] = {}
        for w1, c1 in e1.items():
            for w2, c2 in e2.items():
                fwd, rev = w1 + w2, w2 + w1
                out[fwd] = out.get(fwd, 0) + c1 * c2
                out[rev] = out.get(rev, 0) - c1 * c2
        return {w: c for w, c in out.items() if c != 0}

    def _deg(h) -> int:
        return 1 if isinstance(h, str) else _deg(h[0]) + _deg(h[1])

    # Build Hall set level by level.
    # Ordering: all degree-k elements < all degree-(k+1) elements.
    # Within a degree, letters use channel order; brackets [h1,h2] are
    # sorted by (h1, h2) where each is compared by its position in the
    # accumulated `hall` list (i.e. by index).
    hall: list = list(channels)  # level 1: letters in channel order
    idx = {ch: i for i, ch in enumerate(channels)}  # element → position

    for deg in range(2, level + 1):
        new = []
        for d1 in range(1, deg):
            d2 = deg - d1
            left_elems  = [h for h in hall if _deg(h) == d1]
            right_elems = [h for h in hall if _deg(h) == d2]
            for h1 in left_elems:
                for h2 in right_elems:
                    if idx[h1] >= idx[h2]:      # need h1 < h2
                        continue
                    if not isinstance(h2, str): # if h2 is a bracket [h21, h22]
                        h21 = h2[0]
                        if idx[h1] < idx[h21]:  # need h1 >= h21
                            continue
                    new.append((h1, h2))
        # Sort new brackets by (h1_index, h2_index) to match esig ordering
        new.sort(key=lambda b: (idx[b[0]], idx[b[1]]))
        for h in new:
            idx[h] = len(hall)
            hall.append(h)

    # ── Step 3: extract Hall coordinates via linear solve ────────────
    # At each degree d, build the expansion matrix M where M[i,j] is the
    # coefficient of tensor-word i in the expansion of Hall element j.
    # Then coords = M⁺ · tensor_log_vec  (pseudo-inverse, precomputed once).
    # This handles coupled Hall elements (e.g. [B,[A,C]] and [C,[A,B]]
    # which share tensor words due to the Jacobi identity).
    result: dict[Tuple[str, ...], pl.Expr] = {}

    for deg in range(1, level + 1):
        hall_deg = [h for h in hall if _deg(h) == deg]
        words_deg = [w for w in product(channels, repeat=deg)]

        if deg == 1:
            # Level 1: identity mapping
            for h in hall_deg:
                result[_foliage(h)] = tensor_log[_foliage(h)]
            continue

        # Build expansion matrix and solve for the coordinate transform
        expansions = [_expand(h) for h in hall_deg]
        word_to_idx = {w: i for i, w in enumerate(words_deg)}
        M = np.zeros((len(words_deg), len(hall_deg)))
        for j, exp in enumerate(expansions):
            for w, c in exp.items():
                M[word_to_idx[w], j] = c

        # Pseudo-inverse: coords = M⁺ @ tensor_log_vector
        # M⁺ is (n_hall x n_words), so each Hall coordinate is a fixed
        # linear combination of tensor_log entries — purely expression algebra.
        M_pinv = np.linalg.pinv(M)

        for j, h in enumerate(hall_deg):
            expr = pl.lit(0.0)
            for i, w in enumerate(words_deg):
                coeff = M_pinv[j, i]
                if abs(coeff) < 1e-14:
                    continue
                expr = expr + pl.lit(coeff) * tensor_log[w]
            result[_foliage(h)] = expr

    return result


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
    import sktime
    assert sktime.__version__ == "0.40.1"
    assert iisignature.__version__ == "0.24"
    
    
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

            fast_rolling = df.select(**{",".join(k) + f"_{w}": v for w in window_size for k, v in rolling(sigs, window_size=w).items()})#

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

    ############################################################################
    # LOG-SIGNATURE TESTS
    ############################################################################

    ### TEST LOG-SIGNATURE: component count matches esig
    for lvl in (2, 3):
        _sigs = signature(*map(to_diff, 'ABC'), level=lvl)
        _logsigs = log(_sigs)
        expected = len(esig.stream2logsig(np.zeros((2, 3)), depth=lvl))
        assert len(_logsigs) == expected, f"depth {lvl}: got {len(_logsigs)}, expected {expected}"
    print("LOG-SIG component count tests passed")

    ### TEST LOG-SIGNATURE vs esig at depth 2 & 3
    df = pl.DataFrame({
        "time": pl.date_range(start=dt.date(2025, 1, 1), end=dt.date(2025, 1, 4), eager=True),
        "A": [0, 1, 1, 0],
        "B": [0, 0, 1, 1],
        "C": [1, 2, 3, 2]
    })
    for lvl in (2, 3):
        _sigs = signature(*map(to_diff, 'ABC'), level=lvl)
        log_feats = {",".join(k): v.tail(1) for k, v in log(_sigs).items()}
        numpy.testing.assert_array_almost_equal(
            esig.stream2logsig(df.drop('time').to_numpy().astype(float), depth=lvl),
            df.drop('time').select(**log_feats).row(0),
        )
    print("LOG-SIG vs esig passed (depth 2 & 3)")

    ### TEST LOG-SIGNATURE vs sktime
    logsig_transform = SignatureTransformer(
        augmentation_list=("basepoint",), window_name="global",
        window_depth=None, window_length=None, window_step=None,
        rescaling=None, sig_tfm="logsignature", depth=2,
    )
    df_bp = pl.DataFrame({
        "A": [0, 0, 1, 1, 0], "B": [0, 0, 0, 1, 1], "C": [0, 1, 2, 3, 2]
    })
    _sigs = signature(*map(to_diff, 'ABC'), level=2)
    log_feats_bp = {",".join(k): v.tail(1) for k, v in log(_sigs).items()}
    numpy.testing.assert_array_almost_equal(
        logsig_transform.fit_transform(df_bp.to_numpy()),
        df_bp.select(**log_feats_bp).to_numpy(),
    )
    print("LOG-SIG vs sktime passed")

    ### TEST LOG-SIGNATURE composability: log(rolling(sig, full_window)) == log(sig)
    df = pl.DataFrame({
        "time": pl.date_range(start=dt.date(2025, 1, 1), end=dt.date(2025, 1, 4), eager=True),
        "A": [0, 1, 1, 0], "B": [0, 0, 1, 1], "C": [1, 2, 3, 2]
    })
    _sigs = signature(*map(to_diff, 'ABC'), level=2)
    global_log_feats = {",".join(k): v.tail(1) for k, v in log(_sigs).items()}
    rolling_log_feats = {",".join(k): v.tail(1) for k, v in log(rolling(_sigs, window_size=len(df))).items()}
    numpy.testing.assert_array_almost_equal(
        df.drop('time').select(**global_log_feats).row(0),
        df.drop('time').select(**rolling_log_feats).row(0),
    )
    print("LOG-SIG composability passed: log(rolling(sig)) works")

    ### TEST ROLLING LOG-SIGNATURE vs sktime SignatureTransformer(window_name="sliding")
    # Use raw data (no basepoint row); sktime prepends it via augmentation_list=("basepoint",)
    # Our code uses df_bp (with basepoint prepended manually) to match
    df_raw = pl.DataFrame({"A": [0, 1, 1, 0], "B": [0, 0, 1, 1], "C": [1, 2, 3, 2]})
    df_bp = pl.DataFrame({
        "A": [0, 0, 1, 1, 0], "B": [0, 0, 0, 1, 1], "C": [0, 1, 2, 3, 2]
    })
    for lvl in (2, 3):
        for w in (2, 3):
            logsig_sliding = SignatureTransformer(
                augmentation_list=("basepoint",), window_name="sliding",
                window_length=w, window_step=1,
                rescaling=None, sig_tfm="logsignature", depth=lvl,
            )
            # sktime adds basepoint → sees len(df_raw)+1 rows → (len(df_raw)+1 - w + 1) windows
            sktime_out = logsig_sliding.fit_transform(df_raw.to_numpy())
            n_windows = len(df_raw) + 1 - w + 1  # +1 for basepoint row sktime adds

            _sigs = signature(*map(to_diff, 'ABC'), level=lvl)
            _logsigs = log(rolling(_sigs, window_size=w))
            n_logsig = len(_logsigs)
            ours = df_bp.select(**{",".join(k): v for k, v in _logsigs.items()}).to_numpy()

            sktime_matrix = sktime_out.to_numpy().reshape(n_windows, n_logsig)
            numpy.testing.assert_array_almost_equal(
                sktime_matrix,
                ours[w - 1:],
            )
    print("ROLLING LOG-SIG vs sktime passed (depth 2 & 3, windows 2 & 3)")

    ### TEST ROLLING LOG-SIGNATURE on larger random data vs sktime
    np.random.seed(42)
    for num_col, size in [(3, 20), (5, 20)]:
        raw = np.random.rand(size, num_col)
        raw_bp = np.vstack([np.zeros((1, num_col)), raw])  # basepoint
        df_bp = pl.from_numpy(raw_bp)
        cols = df_bp.columns
        for lvl in (2, 3):
            for w in (2, 5):
                logsig_sliding = SignatureTransformer(
                    augmentation_list=("basepoint",), window_name="sliding",
                    window_length=w, window_step=1,
                    rescaling=None, sig_tfm="logsignature", depth=lvl,
                )
                # sktime adds basepoint → sees size+1 rows → (size+1 - w + 1) windows
                sktime_out = logsig_sliding.fit_transform(raw)
                n_windows = size + 1 - w + 1

                _sigs = signature(*map(to_diff, cols), level=lvl)
                _logsigs = log(rolling(_sigs, window_size=w))
                n_logsig = len(_logsigs)
                ours = df_bp.select(**{",".join(k): v for k, v in _logsigs.items()}).to_numpy()

                sktime_matrix = sktime_out.to_numpy().reshape(n_windows, n_logsig)
                numpy.testing.assert_array_almost_equal(sktime_matrix, ours[w - 1:])
        print(f"  ROLLING LOG-SIG vs sktime passed for shape=({size},{num_col})")

    ### TEST DIMENSIONALITY REDUCTION
    for d in (2, 3, 5, 10):
        for lvl in (2, 3):
            _sigs = signature(*map(to_diff, [f'c{i}' for i in range(d)]), level=lvl)
            assert len(log(_sigs)) < len(_sigs)
    print("LOG-SIG dimensionality reduction verified")

    print("ALL LOG-SIGNATURE TESTS PASSED")
