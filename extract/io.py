from typing import Tuple, AnyStr, Optional
import polars as pl
import polars.selectors as cs
import yfinance_pl as yf
import urllib.request
import zipfile
import io
import json
import dvc.api

from .cache import daily_cache
from . import execute_and_pass

dvc_params = execute_and_pass(dvc.api.params_show)

#rich_df = import_from_gist('b0d32072f234fba73650eb4b1e9c0017', name='rich_df', files_head_member='rich_display_pandas_dataframe.py')

def dzip(url: str, csv_filename_in_zip: str = None) -> io.StringIO:
    with urllib.request.urlopen(url) as response:
        zip_data = response.read()
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        with zf.open(csv_filename_in_zip or zf.namelist()[0]) as csv_file:
            content = csv_file.read().decode('utf-8')
    return io.StringIO(content)


def dump(df: pl.DataFrame, output_path: str, key=['date'], **kwargs):  # TODO : DQ checks should go there, with patito or dataframely
    import warnings
    if output_path is None:
        return
    if set(df.schema.values()) != {pl.Date, pl.Float64} and not key:
        warnings.warn("DataFrame schema does not match expected values. Expected all columns to be either Date or Float64.")
    if key:
        df = df.unique(subset=key).sort(by=key, **kwargs).set_sorted('date')
    print("\n\n", output_path, "\n\n")
    print(df.head(5))
    with open(output_path.replace('data/', 'metrics/').replace(".parquet", ".json"), 'w') as f:  # TODO write to stderr and redict
        json.dump(df.select( # METRICS FOR DVC
            min=pl.struct(pl.col.date.dt.to_string().min()),
            std=pl.struct(cs.numeric().std()), 
            null_ratios=pl.struct(pl.all().null_count().truediv(pl.len())),
            zero_ratios=pl.struct(cs.numeric().sub(0.0).abs().lt(0.001).sum().truediv(pl.len())),
            n_points=pl.len(),
        ).to_dicts()[0], f, indent=2)
    df.write_parquet(output_path)

@daily_cache
def get_stock(ticker: AnyStr) -> pl.DataFrame:
    return yf.Ticker(ticker).history(period='10y', interval='1d')\
        .select(pl.lit(ticker).alias('ticker'), pl.col.date.dt.date(), 'close.amount')\
        .sort(by='date')

def get_stocks(tickers: Tuple[AnyStr]) -> pl.DataFrame:
    res = list()
    for ticker in tickers:
        try:
            _df = get_stock(ticker)
        except RuntimeError:
            print(f"Could not download {ticker}, retry in a moment")
            continue
        res.append(_df)
    return pl.concat(res, how='align_full')

@dvc_params
@daily_cache
def execute_polars(source: str, **params):
    from functools import reduce
    _locals = {'cs': cs, 'pl': pl, 'yf': yf, 'get_stocks': get_stocks, 'reduce': reduce, 'dzip': dzip, **params}
    return eval(source, _locals)

@dvc_params
def get_sources_config(names: Optional[Tuple[AnyStr]] = None, **params):
    if names is None:
        return params['sources']
    return {k: v for k, v in params['sources'].items() if k in names}

@daily_cache
def get_sources(sources: Tuple[str]) -> dict[str, pl.DataFrame]:
    _sources = dict()
    config = get_sources_config(names=sources)
    for name, _map in config.items():
        _sources[name] = execute_polars(_map['polars'])
    return pl.concat(_sources.values(), how='align_full')
