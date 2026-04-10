"""
https://maxhalford.github.io/blog/python-daily-cache/
"""
import datetime as dt
from joblib import Memory

memory = Memory('joblib_cache', verbose=0, compress=True)

def daily_cache_validation_callback(metadata):
    last_call_at = dt.datetime.fromtimestamp(metadata['time'])
    return last_call_at.date() == dt.date.today()

# TODO : we have information about the cadence of each source in the params.yaml file -> create a daily_cache and a monthly_cache
daily_cache = memory.cache(cache_validation_callback=daily_cache_validation_callback)
