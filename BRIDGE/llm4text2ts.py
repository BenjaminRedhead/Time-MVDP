# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

############################################################ 
#  part of code from https://github.com/AdityaLab/lstprompt/blob/main/
############################################################

import os
import torch
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai
import ast

os.environ['OPENAI_API_KEY'] = "xxx"

openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

from lstprompt.models.utils import grid_iter
from lstprompt.models.promptcast import get_promptcast_predictions_data
from lstprompt.models.darts import get_arima_predictions_data
from lstprompt.models.llmtime import get_llmtime_predictions_data
from lstprompt.models.validation_likelihood_tuning import get_autotuned_predictions_data
from lstprompt.data.serialize import serialize_arr, deserialize_str, SerializerSettings
from lstprompt.data.small_context import get_datasets

from sklearn.metrics import mean_squared_error, mean_absolute_error
from ts_metrics import TimeSeriesMetrics

def plot_preds(train, test, pred_dict, model_name, show_samples=False):
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train)
    plt.plot(test, label='Truth', color='black')
    plt.plot(pred, label=model_name, color='purple')
    # shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color='purple')
    if show_samples:
        samples = pred_dict['samples']
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.legend(loc='upper left')
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()


def plot_preds2(train, test, pred_dict, model_name, show_samples=False, plot_median=False, show_figure=False):
    pred_median = pred_dict['median']
    pred_median = pd.Series(pred_median, index=test.index)
    
    pred_mean = pred_dict['mean']
    pred_mean = pd.Series(pred_mean, index=test.index)
    
    mse_median = mean_squared_error(test, pred_median)
    mae_median = mean_absolute_error(test, pred_median)
    mse_mean = mean_squared_error(test, pred_mean)
    mae_mean = mean_absolute_error(test, pred_mean)
    
    ts_metrics = TimeSeriesMetrics(pred_mean, test)
    correlation = ts_metrics.compute_correlation()
    cosine_similarity = ts_metrics.compute_cosine_similarity()
    js_distance = ts_metrics.compute_js_distance()
    
    metrics = {
        'MSE_Median': mse_median,
        'MAE_Median': mae_median,
        'MSE_Mean': mse_mean,
        'MAE_Mean': mae_mean,
        'Correlation': correlation,
        'Cosine_Similarity': cosine_similarity,
        'JS_Distance': js_distance
    }
    
    # Print metrics
    for metric, value in metrics.items():
        print(f'{metric}: {value:.2f}')
    
    if show_figure:
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(train, label='Train', color='blue')
        plt.plot(test, label='Truth', color='black')
        if plot_median:
            plt.plot(pred_median, label=f'Pred_Median (MSE: {mse_median:.2f}, MAE: {mae_median:.2f})', color='purple')
        plt.plot(pred_mean, label=f'Pred_Mean (MSE: {mse_mean:.2f}, MAE: {mae_mean:.2f}, Corr: {correlation:.2f}, CosSim: {cosine_similarity:.2f}, JS: {js_distance:.2f})', color='red')

        samples = pred_dict['samples']
        lower = np.quantile(samples, 0.05, axis=0)
        upper = np.quantile(samples, 0.95, axis=0)
        plt.fill_between(pred_median.index, lower, upper, alpha=0.3, color='purple')
        
        if show_samples:
            samples = samples.values if isinstance(samples, pd.DataFrame) else samples
            for i in range(min(10, samples.shape[0])):
                plt.plot(pred_median.index, samples[i], color='purple', alpha=0.3, linewidth=1)
        
        plt.legend(loc='upper left')
        
        if 'NLL/D' in pred_dict:
            nll = pred_dict['NLL/D']
            if nll is not None:
                plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.show()
    
    return metrics


def validate_input(train_length, test_length, train_data, test_data):
    assert isinstance(train_length, int) and train_length > 0, "train_length should be a positive integer"
    assert isinstance(test_length, int) and test_length > 0, "test_length should be a positive integer"
    assert isinstance(train_data, np.ndarray) and len(train_data) == train_length, "train_data should be a numpy array of length train_length"
    assert isinstance(test_data, np.ndarray) and len(test_data) == test_length, "test_data should be a numpy array of length test_length"


def safe_convert_to_float(series):
    if isinstance(series, str): 
        try:
            series = ast.literal_eval(series)
        except (ValueError, SyntaxError) as e:
            print(f"Error converting series: {series}, error: {e}")
            return None 
    
    if isinstance(series, (list, tuple)): 
        return [el if isinstance(el, (int, float, list, tuple)) else None for el in series]
    
    elif isinstance(series, (int, float)): 
        return series
    
    else:
        print(f"Unexpected type for series: {series}, type: {type(series)}")
        return None  


def convert_to_series(data):
    for i in range(len(data)):
        if isinstance(data[i], float):
            data[i] = pd.Series([data[i]], index=pd.RangeIndex(start=0, stop=1, step=1))

        elif isinstance(data[i], (list, tuple, pd.Series)):
            try:
                data[i] = pd.Series(data[i], index=pd.RangeIndex(start=0, stop=len(data[i]), step=1))
            except Exception as e:
                print(f"Error converting data[{i}] to Series: {e}")
        else:
            print(f"Skipping data[{i}] as it is not iterable: {data[i]} (type: {type(data[i])})")
