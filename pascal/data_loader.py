from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Union
from os.path import isfile

class DataSet:
    
    def __init__(self, data, attrs, data_file_path:str="", attrs_file_path:str="") -> None:
        self.data = data
        self.attrs = attrs
        self.data_file_path = data_file_path
        self.attrs_file_path = attrs_file_path
        
    @staticmethod
    def from_csv_file(data_file:str, attrs_file:str) -> DataSet:
        return open_csv_file(data_file, attrs_file=attrs_file)
    
    def fill_undefined_data_points(self, columns:list=None, missing_symbol:str="?", method:str="mean", copy=False) -> Union[None,DataSet]:
        """Fills in undefined data points in columns

        Args:
            columns (list, optional): List of columns to replace undefined points for, 
            if None all columns are checked. Defaults to None.
            missing_symbol (str, optional): Keyword to look for when a value is undefined. Defaults to "?".
            method (str, optional): Replacement method. Defaults to "mean".
            copy (bool, optional): Return a new copy of the DataSet object. Defaults to False.

        Returns:
            Union[None,DataSet]: If copy arg is False, applies changes to current 
            instance of the data else returns a new instance of the class with transformation applied
        """


def _parse_attrs_file(file_path:str):
    if not isfile(file_path):
        raise ValueError("attrs file path is invalid")
    
    attrs = {}
    with open(file_path, "r") as attrs_file:
        lines = attrs_file.readlines()
        for line in lines:
            line = line.strip()
            col_name = line.split(":")[0]
            col_vals = line.split(":")[1].split(",")
            attrs[col_name] = col_vals
            
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
        col_names = attrs.keys()
            
    
    data = pd.read_csv(file,
        sep=delimiter,
        header=0 if (contains_header and col_names is None) else None,
        names=col_names
    )
    
    return DataSet(data, attrs, data_file_path=file, attrs_file_path=attrs_file)