import pandas as pd
import numpy as np



# take natual log, then scale
def is_DF(df):
    return type(df) is pd.DataFrame


def mm_df(df, v_min=None, v_max=None):
    df = df.copy()
    _v = df.to_numpy() if is_DF(df) else df
    v_min = v_min if v_min else _v.min()
    v_max = v_max if v_max else _v.max()
    df = (df - v_min) / (v_max - v_min)
    return df, v_min, v_max


def inv_mm_df(df, v_min, v_max):
    df = df.copy()
    df = df * (v_max - v_min) + v_min
    return df


def log_df(df, scale=None, v_max=None):
    df = df.copy()
    _v = df.to_numpy() if is_DF(df) else df
    v_sc = scale if scale else np.mean(_v[_v!=0]) / 10
    df = df / v_sc
    df = np.log1p(df)
    _v = df.to_numpy() if is_DF(df) else df
    v_max = v_max if v_max else _v.max()
    df = df / v_max
    return df, v_sc, v_max


def inv_log_df(df, v_sc, v_max):
    df = df.copy()
    df = df * v_max
    df = np.expm1(df)
    df = df * v_sc
    return df

