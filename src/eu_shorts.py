import polars as pl
from common.io import dump

shorts = pl.read_csv(
    'https://www.data.gouv.fr/fr/datasets/r/c2539d1c-8531-4937-9cba-3bd8e9786cc5',
    separator=';',
    try_parse_dates=True,
).drop('Date de debut position') # start_date is the date at which we have the information -> mimic production environment.
columns = {
    'Detenteur de la position courte nette': 'Holder',
    'Emetteur / issuer': 'Issuer',
    'code ISIN': 'ISIN',
    #'Date de debut position': 'start_date',
    'Date de fin de publication position': 'end_date',
    'Date de debut de publication position': 'date',
}

shorts = shorts.rename(columns)
#shorts = shorts.pivot('Holder', index=['ISIN', 'start_date', 'end_date'], aggregate_function='sum', values='Ratio')
 # TODO : add rolling gini index ?
shorts = shorts.group_by('ISIN', 'date').agg(short_pos_sum=pl.col.Ratio.sum(), short_pos_median=pl.col.Ratio.median(), short_pos_max=pl.col.Ratio.max())
dump(shorts, key=['ISIN', 'date'], descending=False)
