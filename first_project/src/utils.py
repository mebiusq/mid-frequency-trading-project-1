import numpy as np
import pandas as pd
import statsmodels.api as sm

def calculate_half_life(spread): #cчитаем коинтеграцию, причем нам нужна не просто нужна она, а как быстро спред вернется к 0.
    lag = spread.shift(1)
    ret = spread - lag

    df_temp = pd.DataFrame({'ret': ret, 'lag': lag}).dropna()
    if df_temp.empty:
        return np.inf

    ret_clean = df_temp['ret']
    lag_clean = df_temp['lag']

    if lag_clean.nunique() < 2:
        return np.inf

    X = sm.add_constant(lag_clean)
    try:
        mdl = sm.OLS(ret_clean, X).fit()
    except:
        return np.inf

    if len(mdl.params) < 2:
        return np.inf

    theta = -mdl.params[1]
    if theta <= 0:
        return np.inf
    return -np.log(0.5) / theta