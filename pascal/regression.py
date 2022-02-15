from __future__ import annotations

import pandas as pd
import numpy as np

from pascal.data_loader import DataSet
from abc import ABC, abstractmethod

class Regressor(ABC):
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def fit(self, ds:DataSet):
        self.dataset = ds
        pass
    
    @abstractmethod
    def predict(self, X:DataSet|pd.DataFrame):
        pass
    
    def fit_predict(self, train:DataSet, test:DataSet|pd.DataFrame):
        self.fit(train)
        return self.predict(test)


class AverageValueRegressor(Regressor):
    
    def __init__(self):
        super().__init__()
        
    def fit(self, ds: DataSet, y_col:str=None):
        super().fit(ds)
        
        if y_col is None:
            y_col = ds.attrs.labels_col_name
            
        if y_col is None:
            raise ValueError("Labels column could not be determined")
        
        outputs = ds.data[y_col]
        
        self.mean_output = outputs.mean()
    
    def predict(self, X:DataSet|pd.DataFrame):
        if type(X) == DataSet:
            X = X.get_x()
        return np.full(X.shape[0], self.mean_output)