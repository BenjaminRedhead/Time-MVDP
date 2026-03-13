# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance, jensenshannon
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.seasonal import STL

class TimeSeriesMetrics:
    def __init__(self, predictions, ground_truth):
        self.predictions = np.array(predictions)
        self.ground_truth = np.array(ground_truth)

    def compute_mse(self):
        return mean_squared_error(self.ground_truth, self.predictions)

    def compute_mae(self):
        return mean_absolute_error(self.ground_truth, self.predictions)

    def compute_correlation(self):
        if len(self.predictions) != len(self.ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        mean_predictions = np.mean(self.predictions)
        mean_ground_truth = np.mean(self.ground_truth)
        
        numerator = np.sum((self.predictions - mean_predictions) * (self.ground_truth - mean_ground_truth))
        denominator = np.sqrt(np.sum((self.predictions - mean_predictions)**2) * np.sum((self.ground_truth - mean_ground_truth)**2))
        
        if denominator == 0:
            raise ValueError("Denominator in correlation calculation is zero, cannot divide by zero")
        
        correlation = numerator / denominator
        return correlation

    def compare_distributions(self, interval=1, bar_width=0.2, plot=True):
        min_value = min(np.min(self.predictions), np.min(self.ground_truth))
        max_value = max(np.max(self.predictions), np.max(self.ground_truth))
        
        bins = np.arange(min_value, max_value + interval, interval)
        
        pred_hist, _ = np.histogram(self.predictions, bins=bins, density=True)
        gt_hist, _ = np.histogram(self.ground_truth, bins=bins, density=True)
        
        if plot:
            plt.figure(figsize=(10, 5))
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            plt.bar(bin_centers - bar_width/2, pred_hist, width=bar_width, alpha=0.5, label='Predictions')
            plt.bar(bin_centers + bar_width/2, gt_hist, width=bar_width, alpha=0.5, label='Ground Truth')
            plt.legend(loc='upper right')
            plt.title('Distribution Comparison')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.show()
        
        return pred_hist, gt_hist

    def compute_cosine_similarity(self):
        pred_reshaped = self.predictions.reshape(1, -1)
        gt_reshaped = self.ground_truth.reshape(1, -1)
        return cosine_similarity(pred_reshaped, gt_reshaped)[0][0]

    def compute_js_distance(self):
        pred_hist, gt_hist = self.compare_distributions(plot=False)
        return jensenshannon(pred_hist, gt_hist)

    def compute_ks_test(self):
        """
        Kolmogorov-Smirnov test for the equality of distribution.
        Returns the KS statistic and p-value.
        """
        statistic, p_value = ks_2samp(self.predictions, self.ground_truth)
        return {"ks_statistic": statistic, "p_value": p_value}

    def compute_wasserstein_distance(self):
        """
        Compute the first Wasserstein distance (also called Earth Mover’s Distance).
        """
        return wasserstein_distance(self.predictions, self.ground_truth)

    @staticmethod
    def perform_stl_decomposition(time_series, period):
        if not isinstance(time_series, pd.Series):
            raise ValueError("Input must be a pandas Series")
        if not isinstance(time_series.index, pd.DatetimeIndex):
            raise ValueError("Input Series must have a DatetimeIndex")

        stl = STL(time_series, period=period)
        result = stl.fit()

        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid

        decomposed = pd.DataFrame({
            'original': time_series,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        })

        return decomposed

    @staticmethod
    def load_and_prepare_data(file_path, column_name='history_data', num_steps=96):
        df = pd.read_csv(file_path)
        df[column_name] = df[column_name].apply(ast.literal_eval)
        time_series = pd.Series(df[column_name].iloc[0][:num_steps])
        
        date_range = pd.date_range(start='2021-01-01', periods=num_steps, freq='H')
        time_series.index = date_range
        
        print(time_series)
        return time_series
