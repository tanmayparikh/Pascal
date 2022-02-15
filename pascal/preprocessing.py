from __future__ import annotations

import pandas as pd

from pascal.data_loader import DataSet
from pandas.api.types import is_numeric_dtype

def _cut_to_bins(df:pd.DataFrame, col:str, bins:list[float]) -> list[pd.DataFrame]:
    """Cuts the data into the bin ranges provided

    Args:
        df (pd.DataFrame): Input Data
        col (str): Name of column to use for splitting
        bins (list[float]): List of bin boundaries

    Returns:
        list[pd.DataFrame]: List of data split into respective bins
    """
    ret_dfs = []
    nbins = len(bins)
    for i in range(nbins):
        if i == 0:
            ret_dfs.append(df[df[col] < bins[1]])
        elif i == (nbins - 1):
            ret_dfs.append(df[df[col] >= bins[i]])
        else:
            ret_dfs.append(df[(df[col] >= bins[i]) & (df[col] < bins[i+1])])
    
    return ret_dfs     

def equal_width_discretization(data_set:pd.DataFrame|DataSet, col:str, nbins:int) -> tuple[list[pd.DataFrame], list[float]]:
    """Splits the data into equal width dicrete bins using given column and
    number of bins

    Args:
        data_set (pd.DataFrame | DataSet): Input Data
        col (str): Column name to use for discretization
        nbins (int): Number of bins

    Returns:
        list[pd.DataFrame]: Array of data bins
    """
    if type(data_set) == DataSet:
        data_set = data_set.data
    
    col_data = data_set[col]
    
    data_min = col_data.min()
    data_max = col_data.max()
    bin_width = (data_max - data_min) / nbins
    
    bin_ranges = [data_min + (bin_width * i) for i in range(nbins)]
    
    return _cut_to_bins(data_set, col, bin_ranges), bin_ranges

def equal_fequency_discretization(data_set:pd.DataFrame|DataSet, col:str, nbins:int) -> tuple[list[pd.DataFrame], list[float]]:
    """Splits the data into n bins, but each bin contains approximately the same
    number of samples

    Args:
        data_set (pd.DataFrame | DataSet): Input Data
        col (str): Column name to use for discretization
        nbins (int): Number of bins

    Returns:
        list[pd.DataFrame]: Array Of ddata bins
    """
    if type(data_set) == DataSet:
        data_set = data_set.data
    
    col_data = data_set[col]
    
    _, bin_ranges = pd.qcut(col_data, nbins, retbins=True, duplicates="drop")
    bin_ranges = bin_ranges.tolist()
    
    return _cut_to_bins(data_set, col, bin_ranges), bin_ranges

def standardize(data_set:pd.DataFrame, cols:None|str|list[str] = None) -> pd.DataFrame:
    """Applies z-score standardization to selected columns

    Args:
        data_set (pd.DataFrame): Input Data
        cols (None | str | list[str], optional): Columns to standardize, single column, list of columns or
        None for all columns. Defaults to None.

    Returns:
        pd.DataFrame: [description]
    """
    
    data = data_set.copy()
    
    if cols is None:
        cols = data.columns
    elif isinstance(cols, str):
        cols = [cols]
    
    for col in cols:
        if not is_numeric_dtype(data[col]):
            continue
        
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std
    
    return data