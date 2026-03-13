# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import openai
import numpy as np

from lstprompt.models.promptcast import get_promptcast_predictions_data
from lstprompt.models.llmtime import get_llmtime_predictions_data
from lstprompt.models.utils import grid_iter
from lstprompt.models.gpt import gpt_completion_fn
from lstprompt.data.small_context import get_datasets
from lstprompt.data.serialize import SerializerSettings
from lstprompt.models.llms import completion_fns
from functools import partial

class LSTPromptPredictor:
    """
    Predictor class for LSTPrompt-based models.
    Supports LLMTime and PromptCast methods for time series prediction.
    """

    def __init__(self,
                 model_name="gpt-4",
                 dataset_name="AirPassengersDataset",
                 method="LLMTime",
                 num_samples=5,
                 prediction_length=24):
        """
        Initialize the LSTPrompt predictor.

        Args:
            model_name (str): Name of the OpenAI model (e.g., 'gpt-4o-2024-05-13').
            dataset_name (str): Dataset to use (must exist in get_datasets()).
            method (str): Prediction method: 'LLMTime' or 'PromptCast'.
            num_samples (int): Number of samples per prediction.
            prediction_length (int): Forecasting horizon.
        """

        # Set attributes
        self.model_name = model_name if isinstance(model_name, str) else model_name[0]
        self.dataset_name = dataset_name
        self.method = method
        self.num_samples = num_samples
        self.prediction_length = prediction_length

        # Load datasets
        datasets = get_datasets()
        if dataset_name not in datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(datasets.keys())}")

        self.train_data, self.test_data = datasets[dataset_name]

        # Set serializer settings
        self.serializer_settings = SerializerSettings(
            base=10,
            prec=3,
            signed=True,
            fixed_length=False,
            max_val=10000000.0,
            time_sep=' ,',
            bit_sep=' ',
            plus_sign='',
            minus_sign=' -',
            half_bin_correction=True,
            decimal_point='',
            missing_str=' Nan'
        )

        # Register the model into the completion functions if not already
        if self.model_name not in completion_fns:
            print(f"[LSTPromptPredictor] Registering model '{self.model_name}' in completion_fns.")
            completion_fns[self.model_name] = partial(gpt_completion_fn, model=self.model_name)

        print(f"[LSTPromptPredictor] Initialized with model '{self.model_name}', method '{self.method}', dataset '{self.dataset_name}'.")

    def predict(self, train_series=None, description="", verbose=False):
        """
        Predict the time series using the selected method.

        Args:
            train_series (list or np.ndarray): Time series history for training (optional).
            description (str): Optional textual description (not used currently).
            verbose (bool): Print debug information.

        Returns:
            list: Predicted time series.
        """

        if train_series is None:
            train_series = self.train_data


        if self.method.lower() == "llmtime":
            return self._predict_llmtime(train_series, description, verbose)

        elif self.method.lower() == "promptcast":
            return self._predict_promptcast(train_series, description, verbose)

        else:
            raise ValueError(f"Unsupported prediction method: {self.method}")

    def _predict_llmtime(self, train_series, description, verbose=False):
        gpt_hypers = dict(
            temp=0.3,
            alpha=0.95,
            beta=0.7,
            basic=True,
            settings=self.serializer_settings
        )

        model_hypers = {
            'LLMTime GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **gpt_hypers},
        }

        model_predict_fns = {
            'LLMTime GPT-3.5': get_llmtime_predictions_data,
        
        }
        model_names = list(model_predict_fns.keys())

        pred_dict = get_llmtime_predictions_data(
            train_series[:],
            self.test_data,
            model_hypers,
            self.num_samples,
            self.model_name,  
            verbose=verbose,
            parallel=False,
            prompt_method=self.method,
            all_time_steps=len(train_series),
            breath_steps=self.prediction_length,
            dataname=self.dataset_name
        )

        means = self._process_predictions(pred_dict)

        if verbose:
            print("[LSTPromptPredictor] LLMTime Prediction Means:", means)

        return means.tolist()


    def _predict_promptcast(self, train_series, description, verbose=False):
        """
        Predict using PromptCast method.
        """
        promptcast_hypers = dict(
            temp=0.3,
            alpha=0.95,
            beta=0.7,
            basic=True,
            settings=self.serializer_settings
        )

        model_hypers = {'PromptCast GPT-4': {'model': self.model_name, **promptcast_hypers}}
        hyper_list = list(grid_iter(model_hypers['PromptCast GPT-4']))

        if verbose:
            print(f"[LSTPromptPredictor] Running PromptCast with {len(hyper_list)} hyperparameter combinations.")

        predictions = []
        for hypers in hyper_list:
            model = hypers['model']
            if model not in completion_fns:
                raise ValueError(f"Invalid model '{model}'. Available models: {list(completion_fns.keys())}")

            pred_dict = get_promptcast_predictions_data(
                train_series[:],
                self.test_data,
                [hypers],
                self.num_samples,
                verbose=verbose,
                parallel=False
            )
            predictions.append(pred_dict)

        means = self._process_predictions(predictions[0])

        if verbose:
            print(f"[LSTPromptPredictor] PromptCast Prediction Means: {means}")

        return means.tolist()

    def _process_predictions(self, pred_dict):
        """
        Filter outliers and calculate mean predictions.

        Args:
            pred_dict (dict): Output predictions from LSTPrompt models.

        Returns:
            np.ndarray: Mean predictions per time step.
        """
        samples = pred_dict['samples']
        means = {}

        for col in samples.columns:
            series = samples[col]
            std_dev = series.std()
            mean_val = series.mean()

            lower = mean_val - 1.5 * std_dev
            upper = mean_val + 1.5 * std_dev

            filtered = series[(series >= lower) & (series <= upper)]

            means[col] = filtered.mean()

        means_array = np.array(list(means.values()))
        return means_array
