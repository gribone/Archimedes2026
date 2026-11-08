"""Shrink observed customer concentrations toward the registry prevalence."""
import numpy as np
import pandas as pd


SECONDARY_BAND_WEIGHT = .08


def market_affinity(companies, customer_vats, primary_codes):
    # Each VAT contributes one unit distributed over its venues.
    weight = 1 / companies.groupby('vat').vat.transform('size')
    member = companies.vat.isin(customer_vats)
    prior = float(weight[member].sum() / weight.sum())
    activity = pd.Series([min(codes) if codes else '' for codes in primary_codes], index=companies.index)
    region = companies.prefecture_name
    municipality = companies.municipality_name
    signals = [(region, .20), (municipality, .27), (activity, .15),
               (region + '/' + activity, .22), (companies.legal_type, .08),
               (companies.incorporation_date.str[:4], .08)]
    log_lift = np.zeros(len(companies))
    for key, importance in signals:
        total = weight.groupby(key).sum()
        customers = (weight * member).groupby(key).sum()
        # Five prior customer equivalents keep tiny groups close to neutral.
        lift = (customers + 5) / (total * prior + 5)
        log_lift += importance * np.log(key.map(lift).to_numpy())
    if 'secondary_band' in companies:
        # How many secondary activities a business declares (bands): optional, set by the pipeline.
        band = companies.secondary_band
        total = weight.groupby(band).sum()
        customers = (weight * member).groupby(band).sum()
        lift = (customers + 5) / (total * prior + 5)
        log_lift += SECONDARY_BAND_WEIGHT * np.log(band.map(lift).fillna(1.).to_numpy(dtype=float))
    if 'hotel_band' in companies:
        band = companies.hotel_band
        total = weight.groupby(band).sum()
        customers = (weight * member).groupby(band).sum()
        lift = (customers + 5) / (total * prior + 5)
        log_lift += .10 * np.log(band.map(lift).fillna(1.).to_numpy(dtype=float))
    return log_lift
