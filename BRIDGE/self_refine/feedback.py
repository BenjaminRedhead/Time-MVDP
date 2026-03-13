# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import openai
import re
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp, wasserstein_distance
from typing import List, Tuple, Dict
import json

class TimeSeriesFeedback:
    def __init__(self, model: str, input_instance: str = None, max_tokens: int = 512):
        """
        Feedback initialization for time series and description evaluation.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.prompt = ""
        if input_instance:
            self.setup_prompt_from_input(input_instance)

    def evaluate(self, generated_series: List[float], actual_series: List[float], text_description: str) -> Dict:
        """
        Main evaluation function.
        """
        # Numerical metrics
        mse = mean_squared_error(actual_series, generated_series)
        ks_stat, ks_p_value = ks_2samp(generated_series, actual_series)
        wd = wasserstein_distance(generated_series, actual_series)

        # Text quality and suggestions
        text_quality_scores, suggestions, summary_feedback = self.text_feedback(text_description, generated_series)

        result = {
            "mse": mse,
            "ks_stat": ks_stat,
            "ks_p_value": ks_p_value,
            "wasserstein_distance": wd,
            "text_quality_scores": text_quality_scores,
            "suggestions": suggestions,
            "text_feedback_summary": summary_feedback
        }

        return result

    def setup_prompt_from_input(self, examples_path: str):
        """
        Setup prompt from example jsonl for better few-shot prompting.
        """
        template = """Time Series: {time_series}
                Text Description: {text_description}
                Does the description accurately represent the time series?
                Feedback: {feedback}
                """
        examples_df = pd.read_json(examples_path, orient="records", lines=True)
        prompt_parts = []
        for _, row in examples_df.iterrows():
            prompt_parts.append(template.format(
                time_series=row["time_series"],
                text_description=row["text_description"],
                feedback=row.get("feedback", "No feedback provided.")
            ))

        instruction = (
            "You are an expert evaluating time series descriptions. "
            "Provide ratings and suggestions for each of these aspects:\n"
            "1. Accuracy of trend description\n"
            "2. Mention of seasonality\n"
            "3. Reference to external factors\n"
            "4. Clarity of description\n"
            "5. Completeness of information\n"
            "Also, summarize the key feedback points."
        )

        self.prompt = instruction + "\n\n###\n\n" + "\n\n###\n\n".join(prompt_parts)

    def text_feedback(self, description: str, time_series: List[float]) -> Tuple[Dict, Dict, str]:
        """
        Ask OpenAI to provide detailed feedback including suggestions for improvements.
        """
        query = f"""
            Time Series: {time_series}
            Text Description: {description}

            Please provide:
            1. A score for each of the following aspects out of 5: 
            - Accuracy of trend description
            - Mention of seasonality
            - Reference to external factors
            - Clarity of description
            - Completeness of information
            2. For each aspect, write one actionable suggestion to improve the description.
            3. Finally, provide an overall feedback summary.

            Return the result in JSON format like:
            {{
                "scores": {{
                    "accuracy_of_trend": "5/5",
                    "mention_of_seasonality": "3/5",
                    "reference_to_external_factors": "1/5",
                    "clarity_of_description": "4/5",
                    "completeness_of_information": "3/5"
                }},
                "suggestions": {{
                    "accuracy_of_trend": "No changes needed.",
                    "mention_of_seasonality": "Consider mentioning the lack or presence of seasonality.",
                    "reference_to_external_factors": "Mention external factors such as policy or economic events.",
                    "clarity_of_description": "Use more precise language, avoid vague terms.",
                    "completeness_of_information": "Include peak and dip values with timestamps."
                }},
                "summary_feedback": "The description captures the trend well, but lacks references to seasonality and external factors."
            }}
        """

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": query}],
            temperature=0.0,
            max_tokens=self.max_tokens
        )

        content = response['choices'][0]['message']['content']

        try:
            result_json = json.loads(content)
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON: {e}")
            result_json = {
                "scores": {},
                "suggestions": {},
                "summary_feedback": "Could not parse the feedback. Please try again."
            }

        return result_json.get("scores", {}), result_json.get("suggestions", {}), result_json.get("summary_feedback", "")

    def __call__(self, time_series: List[float], text_description: str) -> str:
        """
        Call for simplified feedback.
        """
        prompt = self.make_query(time_series, text_description)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=self.max_tokens
        )

        return response['choices'][0]['message']['content']

    def make_query(self, time_series: List[float], text_description: str) -> str:
        """
        Builds the query to evaluate description quality.
        """
        question = f"""Time Series: {time_series}
            Text Description: {text_description}
            Does the description accurately represent the time series?
            """
        return self.prompt + "\n\n###\n\n" + question
