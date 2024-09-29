from abc import ABC
from dataclasses import dataclass
from gluonts.dataset.common import ListDataset
import pandas as pd

@dataclass
class GluontsDataset(ABC):
    '''
    A dataclass storing train, validation and test data for time series ListDataset from gluonts.
    '''
    train_data: ListDataset
    val_data: ListDataset
    test_data: ListDataset
    data_table: pd.DataFrame

