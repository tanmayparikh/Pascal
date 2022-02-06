from __future__ import annotations

import pandas as pd

from pandas.api.types import is_numeric_dtype
from typing import Union

def equal_width_discretization(data_set: pd.DataFrame, col:str, nbins:int) -> list[pd.DataFrame]:
    """Splits the data into equal width dicrete bins using given column and
    number of bins

    Args:
        data_set (DataSet): Input Data
        col (str): Column name to use for discretization
        nbins (int): Number of bins

    Returns:
        list[pd.DataFrame]: Array of data bins
    """
    col_data = data_set[col]
    
    data_min = col_data.min()
    data_max = col_data.max()
    bin_width = (data_max - data_min) / nbins
    
    bin_ranges = [(data_min + (bin_width * i), data_min + (bin_width * (i + 1))) for i in range(nbins)]
    
    ret_dfs = []
    for i, bin_range in enumerate(bin_ranges):
        if i == 0:
            ret_dfs.append(data_set[data_set[col] < bin_range[1]])
        elif i == (nbins - 1):
            ret_dfs.append(data_set[data_set[col] > bin_range[0]])
        else:
            ret_dfs.append(data_set[(data_set[col] >= bin_range[0]) & (data_set[col] < bin_range[1])])
    
    return ret_dfs

def standardize(data_set: pd.DataFrame, cols:Union[None,str,list[str]] = None) -> pd.DataFrame:
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