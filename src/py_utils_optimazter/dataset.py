from abc import ABC
from dataclasses import dataclass
from gluonts.dataset.common import ListDataset
from pandas import DataFrame, Series

@dataclass
class GluontsDataset(ABC):
    '''
    A dataclass storing train, validation and test data for time series ListDataset from gluonts.
    '''
    train_data: ListDataset
    val_data: ListDataset
    test_data: ListDataset
    data_table: DataFrame



@dataclass
class PandasNormalizer:

    df_min: Series = None
    df_max: Series = None 


    def normalize(self, df: DataFrame)->DataFrame:
        self.df_min = df.min()
        self.df_max = df.max()
        df = (df - self.df_min) / (self.df_max - self.df_min)
        return df
    
    def denormalize(self, df: DataFrame)->DataFrame:
        df = df * (self.df_max - self.df_min) + self.df_min
        return df