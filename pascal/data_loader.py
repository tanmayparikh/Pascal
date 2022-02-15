from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Union
from os.path import isfile
from pathlib import Path

class DataSet:
    
    def __init__(self, data: pd.DataFrame, attrs: DataSetAttributes, 
                 data_file_path:str="", attrs_file_path:str="") -> None:
        self.data = data
        self.attrs = attrs
        self.data_file_path = data_file_path
        self.attrs_file_path = attrs_file_path
        
    @staticmethod
    def from_csv_file(data_file:str, attrs_file:str=None) -> DataSet:
        return open_csv_file(data_file, attrs_file=attrs_file)
    
    def replace_undefined_data_points(self, 
                                   columns:list=None, 
                                   excluded_cols:list=None,
                                   missing_symbol:str="?", 
                                   method:str="mean", 
                                   verbose:bool=False,
                                   copy:bool=False) -> Union[None,DataSet]:
        """Fills in undefined data points in a given row with a missing_symbol identifier

        Args:
            columns (list, optional): List of columns to replace undefined points for, 
            if None all columns are checked
            excluded_cols (list, optional): List of columns to exclude. Defaults to None.
            missing_symbol (str, optional): Keyword to look for when a value is undefined. Defaults to "?".
            method (str, optional): Replacement method. Defaults to "mean".
            verbose (bool, optional): Print debug messages. Defaults to False.
            copy (bool, optional): Return a new copy of the DataSet object. Defaults to False.

        Returns:
            Union[None,DataSet]: If copy arg is False, applies changes to current 
            instance of the data else returns a new instance of the class with transformation applied
        """
        cols = None
        if columns is None:
            cols = self.data.columns
            
        if excluded_cols is not None:
            cols = list(set(cols).difference(set(excluded_cols)))
            
        if verbose:
            cols_str = ", ".join(cols)
            print(f"Replacing undefined points for columns: {cols_str}")
            
        ds = self
        if copy:
            ds = self.make_copy()
            
        for col in cols:
            missing_rows = ds.data[ds.data[col] == missing_symbol][col]
            if missing_rows.shape[0] == 0:
                continue
            
            if verbose:
                print(f"Found {missing_rows.shape[0]} undefined values in col: {col}")
            
            if method == "mean":
                col_type = ds.attrs.get_column_type(col)
                if col_type is None:
                    continue
                
                values = ds.data[ds.data[col] != missing_symbol][col].astype(col_type)
                mean = values.mean().astype(col_type)
                
                ds.data[col].replace(missing_symbol, mean, inplace=True)
                ds.data[col] = ds.data[col].astype(col_type)
                
        if copy:
            return ds
                    
    def ordinal_encode(self, col:str, inplace=False) -> pd.Series:
        """Ordinally encodes given column
        E.g. [A,B,C] -> [0, 1, 2]

        Args:
            col (str): Name of column to encode
            inplace (bool): Change the original column in the dataset

        Returns:
            pd.Series: Encoded values
        """
        col_attrs = self.attrs.get_attr_by_col_name(col)
        vals = self.data[col]
        if col_attrs is None:
            options = vals.unique()
        else:
            options = np.array(col_attrs.choices)
        
        encoded = [np.argwhere(options == val)[0][0] for val in vals]
        
        if not inplace:
            return pd.Series(encoded, name=col)
        else:
            self.data[col] = pd.Series(encoded)
    
    def one_hot_encode(self, col:str = None) -> pd.DataFrame:   
        """One-hot encodes given column where each column represents 
        a unique value inside the original data

        Args:
            col (str, optional): Name of column to encode, if None the labels
            column will be encoded

        Returns:
            pd.DataFrame: Encoded values
        """
        
        if col is None:
            labels_col = self.attrs.labels_col_name
            if labels_col is None:
                raise ValueError("Labels column name could not be determined")
        else:
            labels_col = col
        
        vals = self.data[labels_col]
        unique_vals = vals.unique() 
        encoded = pd.DataFrame()
        for unique_val in unique_vals:
            encoded[unique_val] = [1 if x == unique_val else 0 for x in vals]
        
        return encoded
    
    def summary(self) -> pd.DataFrame:
        out = self.data.copy().drop(self.data.index)
        out["Metric"] = ""
        
        cols = self.data.columns
        dtypes = self.data.dtypes
        
        min = []
        max = []
        mean = []
        std = []
        for col, dtype in zip(cols, dtypes):
            col_data = self.data[col]
            if dtype == "int64":
                min.append(f"{col_data.min():.0f}")
                max.append(f"{col_data.max():.0f}")
                mean.append(f"{col_data.mean():.4f}")
                std.append(f"{col_data.std():.4f}")
            elif dtype == "float":
                min.append(f"{col_data.min():.4f}")
                max.append(f"{col_data.max():.4f}")
                mean.append(f"{col_data.mean():.4f}")
                std.append(f"{col_data.std():.4f}")
            else:
                min.append("NA")
                max.append("NA")
                mean.append("NA")
                std.append("NA")
        
        out.loc[0] = min + ["Min"]
        out.loc[1] = max + ["Max"]
        out.loc[2] = mean + ["Mean"]
        out.loc[3] = std + ["Std"]
        
        out.set_index("Metric", inplace=True)
        
        return out
    
    def get_x(self) -> pd.DataFrame:
        label_col_name = self.attrs.labels_col_name
        if label_col_name is None:
            raise ValueError("Label column couldnt be determined")
        
        cols = list(self.data.columns)
        cols.remove(label_col_name)
        
        return self.data[cols]
    
    def get_y(self) -> pd.Series:
        label_col_name = self.attrs.labels_col_name
        if label_col_name is None:
            raise ValueError("Label column couldnt be determined")
        
        return self.data[label_col_name]
    
    def make_copy(self) -> DataSet:
        return DataSet(self.data.copy(), self.attrs, self.data_file_path, self.attrs_file_path)


class DataSetAttributes:
    
    def __init__(self, col_attrs:list[DataSetAttribute], label_col_name:str):
        self.col_attrs = col_attrs
        self.labels_col_name = label_col_name
        
    def get_column_names(self):
        return [attr.name for attr in self.col_attrs]
    
    def get_column_type(self, col:str):
        for col_attr in self.col_attrs:
            if col_attr.name == col:
                return col_attr.type
        return None

    def get_attr_by_col_name(self, col:str):
        for col_attr in self.col_attrs:
            if col_attr.name == col:
                return col_attr
        return None
class DataSetAttribute:
    
    def __init__(self, name:str, type:str, choices:list) -> None:
        self.name = name
        self.type = type
        self.choices = choices
        

def _parse_attrs_file(file_path:str):
    if not isfile(file_path):
        raise ValueError("attrs file path is invalid")
    
    col_attrs = []
    label_col_name = None
    with open(file_path, "r") as attrs_file:
        lines = attrs_file.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            
            if line.startswith("Label Column Name"):
                label_col_name = line.split(":")[1]
            else:
                props = line.split(":")
                col_name = props[0]
                col_type = props[1]
                col_vals = props[2].split(",")
                
                col_attrs.append(DataSetAttribute(col_name, col_type, col_vals))
            
    return DataSetAttributes(col_attrs, label_col_name)
            

def open_csv_file(file:str, 
                  attrs_file:str=None,
                  contains_header:bool=False, 
                  delimiter:str=",") -> DataSet:
    """Loads a csv file from the given file path

    Args:
        file (str): File path to data file
        attrs_file (str, optional): File containing column attributes. Defaults to "".
        contains_header (bool, optional): If the given file contains column names as first line. Defaults to False.
        delimiter (str, optional): Separator to use when parsing the data. Defaults to ",".

    Raises:
        ValueError: If file paths are invalid

    Returns:
        DataSet: Instance of the DataSet class
    """

    
    if not isfile(file):
        raise ValueError(f"file arg: {file} must be a valid file path")
    
    col_names = None
    attrs = None
    
    if attrs_file is None:
        p = Path(file)
        attrs_file = str(p).replace(p.suffix, ".attrs")

    if isfile(attrs_file):
        attrs = _parse_attrs_file(attrs_file)
        col_names = attrs.get_column_names()
            
    
    data = pd.read_csv(file,
        sep=delimiter,
        header=0 if (contains_header and col_names is None) else None,
        names=col_names
    )
    
    return DataSet(data, attrs, data_file_path=file, attrs_file_path=attrs_file)