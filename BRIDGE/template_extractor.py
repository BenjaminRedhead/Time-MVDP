# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import re
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from llm_agents import ChatLLM
from self_refine.feedback import TimeSeriesFeedback


class TemplateExtractor:
    def __init__(self, llm: ChatLLM, feedback_tool: TimeSeriesFeedback, output_template_file="templates/description_template.json"):
        self.llm = llm
        self.feedback_tool = feedback_tool
        self.output_template_file = output_template_file
        self.templates = []

    def load_final_text_from_refinement_result(self, refinement_json_path: str) -> str:
        """
        Load the `final_text` field from multi_agent_refinement_result.json
        """
        print(f"Loading refinement result from {refinement_json_path}...")
        with open(refinement_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        final_text = data.get("final_text", "")
        if not final_text:
            raise ValueError("No 'final_text' found in the provided JSON.")
        return final_text

    def generate_template_from_final_text(self, final_text: str) -> str:
        """
        Generate a generalized template from the `final_text`
        """
        prompt = (
            "You are an expert in time series data description. Given the following finalized description:\n\n"
            f"{final_text}\n\n"
            "Extract a generalized description template. Replace all specific values like numbers, dates, and category names with placeholders in curly braces, "
            "such as {dataset_name}, {frequency}, {start_date}, {end_date}, {prediction_length}, {min_value}, {max_value}, {mean_value}, {std_value}, {peak_steps}, {dip_steps}, {variability_summary}.\n\n"
            "Return ONLY the generalized template in natural language, no explanation."
        )

        try:
            response = self.llm.generate(prompt)
            if not response or len(response.strip()) < 10:
                raise ValueError("LLM returned an empty or too short response.")

            print("\n Extracted template:\n")
            print(response.strip())

            return response.strip()

        except Exception as e:
            print(f"[ERROR] LLM error during template generation: {e}")
            return ""

    def evaluate_template_candidate(self, candidate: str, dummy_series: List[float]) -> bool:
        """
        Evaluate a single sentence to check if it works as a template for time series descriptions.
        """
        print(f"Evaluating template: {candidate}")

        # Step 1: Replace variables in the candidate template with example values
        test_description = (
            candidate.replace("{dataset_name}", "ETTh1")
                     .replace("{frequency}", "hourly")
                     .replace("{data_description}", "electricity consumption")
                     .replace("{start_date}", "January 1, 2020")
                     .replace("{end_date}", "December 31, 2022")
                     .replace("{prediction_length}", "24")
                     .replace("{min_value}", "100")
                     .replace("{max_value}", "250")
                     .replace("{mean_value}", "175")
                     .replace("{std_value}", "30")
                     .replace("{peak_steps}", "June 15, 2021")
                     .replace("{dip_steps}", "September 10, 2021")
                     .replace("{variability_summary}", "moderate fluctuations")
        )

        # Step 2: Evaluate the text quality with feedback tool
        try:
            feedback_score = self.feedback_tool(dummy_series, test_description)
            print(f"Feedback result: {feedback_score}")
        except Exception as e:
            print(f"Failed evaluation on: {test_description}")
            print(f"[ERROR]: {e}")
            return False

        return True  # You can add stricter rules here if necessary

    def extract_templates_from_refinement_result(self, refinement_json_path: str):
        """
        Full pipeline: loads `final_text` -> generalizes -> evaluates -> saves template.
        """
        final_text = self.load_final_text_from_refinement_result(refinement_json_path)

        # Step 1: Generate generalized template
        template = self.generate_template_from_final_text(final_text)

        if not template:
            print("No valid template generated.")
            return

        # Step 2: Evaluate its quality with a dummy series
        dummy_series = [100, 150, 200, 250, 300, 350, 400, 450, 500]

        if self.evaluate_template_candidate(template, dummy_series):
            self.templates.append(template)
            print(f"Accepted template: {template}")
        else:
            print(f"Template rejected after evaluation.")

        # Step 3: Save templates to JSON
        os.makedirs(os.path.dirname(self.output_template_file), exist_ok=True)
        with open(self.output_template_file, "w", encoding="utf-8") as f:
            json.dump(self.templates, f, indent=2)

        print(f"\n Saved {len(self.templates)} template(s) to {self.output_template_file}")

