from __future__ import annotations
from cgi import test

import pandas as pd
import numpy as np

from pascal.data_loader import DataSet

class KFoldCrossValidation:
    
    def __init__(self, ds: DataSet, labels_col_name:str = None) -> None:
        self.dataset = ds
        
        if labels_col_name is not None:
            self.labels_col_name = labels_col_name
        else:
            self.labels_col_name = ds.attrs.labels_col_name
        
        if self.labels_col_name is None:
            raise ValueError("Labels column name could not be determined")
    
    def split(self, K:int):
        """Splits the dataset into K folds

        Args:
            K (int): Number of folds

        Raises:
            ValueError: If invalid K is specified

        Yields:
            Iterator[tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]]: ((x_train, y_train), (x_test, y_test))
        """
        if K < 2:
            raise ValueError("K must be at least 2")
        
        num_samples = self.dataset.data.shape[0]
        
        splits_size = num_samples // K
        leftover = num_samples % K
        
        indices = np.arange(num_samples)
        
        x_col = list(self.dataset.data.columns)
        x_col.remove(self.labels_col_name)
        y_col = self.labels_col_name
        
        last_stop = 0
        for i in range(K):
            start, stop = last_stop, (i * splits_size) + splits_size + leftover
            last_stop = stop
            
            test_indices = indices[start:stop]
            
            x_test = self.dataset.data.loc[test_indices][x_col]
            y_test = self.dataset.data.loc[test_indices][y_col]
            
            x_train = self.dataset.data.drop(test_indices)[x_col]
            y_train = self.dataset.data.drop(test_indices)[y_col]
            
            yield ((x_train, y_train), (x_test, y_test))