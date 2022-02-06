from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Union
from os.path import isfile

class DataSet:
    
    def __init__(self, data: pd.DataFrame, attrs: list[DataSetAttribute], data_file_path:str="", attrs_file_path:str="") -> None:
        self.data = data
        self.attrs = attrs
        self.data_file_path = data_file_path
        self.attrs_file_path = attrs_file_path
        
    @staticmethod
    def from_csv_file(data_file:str, attrs_file:str) -> DataSet:
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
                attr = ds.get_attribute(col)
                if attr is None:
                    continue
                
                values = ds.data[ds.data[col] != missing_symbol][col].astype(attr.type)
                mean = values.mean().astype(attr.type)
                
                ds.data[col].replace(missing_symbol, mean, inplace=True)
                ds.data[col] = ds.data[col].astype(attr.type)
                
        if copy:
            return ds
                    
    def ordinal_encode(self, col:str) -> pd.Series:
        """Ordinally encodes given column
        E.g. [A,B,C] -> [0, 1, 2]

        Args:
            col (str): Name of column to encode

        Returns:
            pd.Series: Encoded values
        """
        vals = self.data[col]
        unique_vals = vals.unique()
        encoded = [np.argwhere(unique_vals == val)[0][0] for val in vals]
        return pd.Series(encoded, name=col)      
    
    def one_hot_encode(self, col:str) -> pd.DataFrame:   
        """One-hot encodes given column where each column represents 
        a unique value inside the original data

        Args:
            col (str): Name of column to encode

        Returns:
            pd.DataFrame: Encoded values
        """
                        
        vals = self.data[col]
        unique_vals = vals.unique() 
        encoded = pd.DataFrame()
        for unique_val in unique_vals:
            encoded[unique_val] = [1 if x == unique_val else 0 for x in vals]
        
        return encoded
        
    def get_attribute(self, name:str):
        for attr in self.attrs:
            if attr.name == name:
                return attr
        
        return None
    
    def make_copy(self) -> DataSet:
        return DataSet(self.data.copy(), self.attrs, self.data_file_path, self.attrs_file_path)
    

class DataSetAttribute:
    
    def __init__(self, name:str, type:str, choices:list) -> None:
        self.name = name
        self.type = type
        self.choices = choices
        

def _parse_attrs_file(file_path:str):
    if not isfile(file_path):
        raise ValueError("attrs file path is invalid")
    
    attrs = []
    with open(file_path, "r") as attrs_file:
        lines = attrs_file.readlines()
        for line in lines:
            line = line.strip()
            props = line.split(":")
            col_name = props[0]
            col_type = props[1]
            col_vals = props[2].split(",")
            
            attrs.append(DataSetAttribute(col_name, col_type, col_vals))
            
    return attrs
            

def open_csv_file(file:str, 
                  attrs_file:str="",
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
    if isfile(attrs_file):
        attrs = _parse_attrs_file(attrs_file)
        col_names = [attr.name for attr in attrs]
            
    
    data = pd.read_csv(file,
        sep=delimiter,
        header=0 if (contains_header and col_names is None) else None,
        names=col_names
    )
    
    return DataSet(data, attrs, data_file_path=file, attrs_file_path=attrs_file)