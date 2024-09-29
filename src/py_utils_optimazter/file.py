import torch
from torch import nn
from datetime import datetime
import os
import json


DATEFORMAT = '%m_%d_%y_%H_%M_%S'

def save_torch_model(model: nn.Module, name:str = 'model', dir:str = 'data'):
    '''
    Save a pytorch model to the local data directory.
    The model will be saved with todays date.
    '''
    torch.save(model.state_dict(), f'{dir}/{name}_{datetime.now().strftime(DATEFORMAT)}.pt')


def list_to_datetime(date_time_list: list):
    return datetime(month=date_time_list[0], day = date_time_list[1], year=date_time_list[2] + 2000, hour=date_time_list[3], minute=date_time_list[4], second=date_time_list[5])


def load_torch_model(model: nn.Module, date_time: datetime, name:str = 'model', dir:str = 'data'):
    '''
    Load a pytorch model from the local data directory.
    Will only work for files saved with save_torch_model
    '''
    state_dict = torch.load(f'{dir}/{name}_{date_time.strftime(DATEFORMAT)}.pt')
    model.load_state_dict(state_dict=state_dict)

def load_latest_torch_model(model: nn.Module, name:str = 'model', dir:str = 'data'):
    '''
    Find the latest pytorch model saved in the local data directory.
    Will only work for files saved with save_torch_model
    '''
    latest_date = None
    for file in os.listdir(dir):
        if file.endswith('.pt') and file.startswith(name):
            date_time_list =[int(x) for x in file.split('.')[0].split('_')[-6:]]
            date_time = list_to_datetime(date_time_list=date_time_list)
            if latest_date:
                if date_time > latest_date:
                    latest_date = date_time
            else:
                latest_date = date_time

    load_torch_model(model, date_time=latest_date, name=name, dir=dir)
    print(f'Loaded weights from model saved {latest_date.strftime("%d/%m/%Y, %H:%M:%S")}')






def log_run(training_loss_history: list, metrics: dict = None, hyperparameters: dict = None, validation_loss_history: list = None, dir = 'data'):
    if os.path.isfile(f'{dir}/log.json'):
        with open(f'{dir}/log.json', 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {}

    with open(f'{dir}/log.json', 'w') as f:
        log_data[f'run {len(log_data)}'] = {
            'datetime' : datetime.now().strftime(DATEFORMAT),
            'training_loss' : training_loss_history,
            'validation_loss' : validation_loss_history if validation_loss_history else [],
            'metrics' : metrics if metrics else {},
            'hyperparameters' : hyperparameters if hyperparameters else {}
        }
        json.dump(log_data, f)



def load_log(dir = 'data'):
    with open(f'{dir}/log.json') as f:
        log_data = json.load(f)
        return log_data