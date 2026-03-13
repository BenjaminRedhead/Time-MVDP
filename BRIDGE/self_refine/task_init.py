# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List
import pandas as pd
from llm_agents import ChatLLM

class TimeSeriesTaskInit:
    def __init__(self, prompt_examples: str, model: ChatLLM) -> None:
        self.model = model
        self.setup_prompt_from_examples_file(prompt_examples)

    def generate_time_series(self, text_description: str) -> List[float]:
        prompt = self.create_prompt(text_description)
        response = self.model.generate(prompt)
        time_series = self.extract_time_series(response)
        return time_series

    def setup_prompt_from_examples_file(self, instances_path: str) -> None:
        TEMPLATE = """Text Description: {text_description}
Time Series: {time_series}"""

        instance_df = pd.read_json(instances_path, orient="records", lines=True)
        prompt = []
        for _, row in instance_df.iterrows():
            example = TEMPLATE.format(
                text_description=row["text_description"],
                time_series=", ".join(map(str, row["time_series"]))
            )
            prompt.append(example)
        self.prompt_examples = "\n\n###\n\n".join(prompt)

    def create_prompt(self, text_description: str) -> str:
        return f"{self.prompt_examples}\n\n###\n\nText Description: {text_description}\nTime Series:"

    def extract_time_series(self, response: str) -> List[float]:
        time_series_str = response.split("Time Series:")[-1].strip()
        time_series = [float(value) for value in time_series_str.split(',') if value.strip()]
        return time_series

    def __call__(self, text_description: str) -> List[float]:
        return self.generate_time_series(text_description)
