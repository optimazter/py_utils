import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from pandas import Series, DataFrame

def plot_loss(f):
    '''
    Decorator function for training algorithms. 
    The decorated function needs to yield the training loss at each iteration.
    '''
    def wrapper(*args, **kwargs)->list:
        loss_hist = []
        for loss in f(*args, **kwargs):
            loss_hist.append(loss)
            plt.plot(np.arange(len(loss_hist)), loss_hist)
            plt.xlabel('Iteration')
            plt.ylabel('Training Loss')
            plt.draw()
            plt.pause(0.1)
            plt.cla()
            plt.show()
        plt.plot(np.arange(len(loss_hist)), loss_hist)
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.draw()
        return loss_hist
    return wrapper


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