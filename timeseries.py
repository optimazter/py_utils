from accelerate import Accelerator
from rich.progress import Progress
from evaluate import load
from gluonts.time_feature import get_seasonality
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from transformers import PretrainedConfig
from typing import Iterable
import inspect
from utils.utils import plot_loss


from gluonts.time_feature import time_features_from_frequency_str
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    Transformation,
    VstackFeatures,
    RenameFields,
)


def create_transformation(freq: str, config: PretrainedConfig, ndim: int) -> Transformation:
    return Chain(
        [
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=ndim,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
            ),
            RenameFields(
                mapping={
                    FieldName.FEAT_TIME: 'time_features',
                    FieldName.TARGET: 'values',
                    FieldName.OBSERVED_VALUES: 'observed_mask',
                }
            ),
        ]
    )


def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
) -> Transformation:
    assert mode in ['train', 'validation', 'test']

    instance_sampler = {
        'train': ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        'validation': ValidationSplitSampler(min_future=config.prediction_length),
        'test': TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field='values',
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=['time_features', 'observed_mask'],
    )



def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    ndim: int,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: int = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        'past_time_features',
        'past_values',
        'past_observed_mask',
        'future_time_features',
    ]

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        'future_values',
        'future_observed_mask',
    ]

    transformation = create_transformation(freq, config, ndim)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, 'train')

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from all the possible transformed time series, 1 in our case)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)
    
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )



def create_backtest_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    ndim: int,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        'past_time_features',
        'past_values',
        'past_observed_mask',
        'future_time_features',
    ]

    transformation = create_transformation(freq, config, ndim)
    transformed_data = transformation.apply(data)

    # we create a Validation Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, 'validation')

    # we apply the transformations in train mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=True)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    ndim: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        'past_time_features',
        'past_values',
        'past_observed_mask',
        'future_time_features',
    ]

    transformation = create_transformation(freq, config, ndim)
    transformed_data = transformation.apply(data, is_train=False)

    # We create a test Instance splitter to sample the very last
    # context window from the dataset provided.
    instance_sampler = create_instance_splitter(config, 'test')

    # We apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )



@plot_loss
def train_time_series_model(model, train_loader, optimizer, epochs: int, plot = True) -> np.array:

    accelerator = Accelerator()
    device = accelerator.device

    model, optimizer, train_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
    )

    train_losses = []

    with Progress() as progress:
        model.train()

        epochs_task = progress.add_task('[green]Training', total=epochs)
        total_batches = sum(1 for __ in enumerate(train_loader))
        batch_task = progress.add_task('[blue]working on batch', total = total_batches)

        for epoch in range(epochs + 1):  
            train_loss = 0.0
            n_batches = 0   
            optimizer.zero_grad()
            
            progress.reset(batch_task)

            for i, batch in enumerate(train_loader):
                outputs = model(
                    past_time_features=batch['past_time_features'].to(device),
                    past_values=batch['past_values'].to(device),
                    future_time_features=batch['future_time_features'].to(device),
                    future_values=batch['future_values'].to(device),
                    past_observed_mask=batch['past_observed_mask'].to(device),
                    future_observed_mask=batch['future_observed_mask'].to(device),
                )
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()

                train_loss += loss.item() * batch['past_values'].size(0)
                n_batches += 1
                progress.update(batch_task, advance=1)

                yield loss.item()
            
   
            progress.update(epochs_task, advance=1, description='[green]Epoch {}/{} | Training loss {:.5f}  | '.format(epoch + 1, epochs, train_loss / total_batches))
        



def generate_time_series_forecasts(model, val_loader, mean = False)-> list:

    accelerator = Accelerator()
    device = accelerator.device

    forecasts = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            outputs = model.generate(
                past_time_features=batch['past_time_features'].to(device),
                past_values=batch['past_values'].to(device),
                future_time_features=batch['future_time_features'].to(device),
                past_observed_mask=batch['past_observed_mask'].to(device),
            )
            if mean:
                forecasts.append(outputs.sequences.mean(dim = 1).cpu().numpy())
            else:
                forecasts.append(outputs.sequences.cpu().numpy())

    return forecasts


def evaluate_time_series_forecast(forecasts: list, val_data, freq: str, pred_length: int) -> tuple:
    mase_metric = load('evaluate-metric/mase')
    smape_metric = load('evaluate-metric/smape')

    forecast_median = np.median(forecasts, 1).squeeze(0).T

    mase_metrics = []
    smape_metrics = []

    with Progress() as progress:

        validation_task = progress.add_task('[green]Evaluating model', total=len(val_data) - 1)

        for item_id, ts in enumerate(val_data):
            training_data = ts['target'][:-pred_length]
            ground_truth = ts['target'][-pred_length:]
            mase = mase_metric.compute(
                predictions=forecast_median[item_id],
                references=np.array(ground_truth),
                training=np.array(training_data),
                periodicity=get_seasonality(freq),
            )
            mase_metrics.append(mase['mase'])

            smape = smape_metric.compute(
                predictions=forecast_median[item_id],
                references=np.array(ground_truth),
            )
            smape_metrics.append(smape['smape'])

            progress.update(validation_task, advance=1)

        return np.mean(mase_metrics), np.mean(smape_metrics)


def evaluate_time_series_model(model: nn.Module, val_loader, val_data, pred_length: int, freq: str):
    accelerator = Accelerator()
    device = accelerator.device
    model.to(device)
    model.eval()


    mase_metric = load('evaluate-metric/mase')
    smape_metric = load('evaluate-metric/smape')

    mase_result, smape_result = np.empty(pred_length), np.empty(pred_length)


    with (
        torch.no_grad(),
        Progress() as progress
    ):
        n_batches = sum(1 for __ in val_loader) 
        validation_task = progress.add_task('[green]Evaluating model', total = (pred_length) * n_batches)

        for batch in val_loader:
            outputs = model.generate(
                past_time_features=batch['past_time_features'].to(device),
                past_values=batch['past_values'].to(device),
                future_time_features=batch['future_time_features'].to(device),
                past_observed_mask=batch['past_observed_mask'].to(device),
            )
            mean_pred = outputs.sequences.mean(dim=1).cpu().numpy()[0]

            val_data_array = np.array([val_data[i]['target'] for i in range(len(val_data))]).transpose(1, 0)
            ground_truth = val_data_array[-pred_length:]
            training = val_data_array[:-pred_length]

            assert not np.isnan(val_data_array).any()
            assert mean_pred.shape == ground_truth.shape
            assert mean_pred.shape[0] == pred_length


            for i in range(pred_length):
                mase_result[i] = mase_metric.compute(
                    predictions=mean_pred[i], 
                    references=ground_truth[i],
                    training=training,
                    periodicity=get_seasonality(freq),
                )['mase']
                smape_result[i] = smape_metric.compute(
                    predictions=mean_pred[i], 
                    references=ground_truth[i],
                )['smape']

                progress.update(validation_task, advance=1)

    return np.mean(mase_result), np.mean(smape_result)



def arg_is_in_func(func, arg: str):
    return arg in inspect.signature(func).parameters.keys()