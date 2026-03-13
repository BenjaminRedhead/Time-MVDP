# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import pandas as pd

class TimeSeriesFeedbackPrompt:
    prompt: str = """We want to evaluate each text description based on how well it describes the given time series on five qualities: i) accuracy of trend description, ii) mention of seasonality, iii) reference to external factors, iv) clarity of description, v) completeness of information.

            Here are some examples of this scoring rubric:

            Time Series: [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250]

            Text Description: "The data shows a steady increase in values over time, indicating a strong upward trend. There are no visible seasonal patterns or fluctuations."

            Scores:

            * Accuracy of trend description: The description accurately identifies the steady increase in the time series. 5/5
            * Mention of seasonality: The description correctly notes the absence of seasonality in the data. 5/5
            * Reference to external factors: The description does not mention any external factors, which may or may not be relevant. 3/5
            * Clarity of description: The description is clear and easy to understand. 5/5
            * Completeness of information: The description covers the main aspects of the time series but could mention the exact rate of increase. 4/5

            * Total score: 22/25

            Time Series: [30, 32, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365]

            Text Description: "The temperatures rise and fall cyclically throughout the year, demonstrating clear seasonal patterns. There is a gradual increase in temperature during the summer months."

            Scores:

            * Accuracy of trend description: The description accurately identifies the cyclical pattern in the time series. 5/5
            * Mention of seasonality: The description correctly notes the presence of seasonality in the data. 5/5
            * Reference to external factors: The description mentions the seasons, which are relevant external factors. 5/5
            * Clarity of description: The description is clear and easy to understand. 5/5
            * Completeness of information: The description is comprehensive and covers all necessary aspects of the time series. 5/5

            * Total score: 25/25
            """

def timeseries_iterate_prompt_to_json(output_file="./feedback.jsonl", allow_empty_feedback=True):
    prompt = TimeSeriesFeedbackPrompt.prompt
    res = []

    examples = prompt.split("###")
    for example in examples:
        try:
            if not example:
                continue
            example = example.strip()

            time_series_match = re.search(r"Time Series: \[(.*)\]", example)
            if not time_series_match:
                continue
            time_series = time_series_match.group(1)
            time_series_list = [float(x) for x in time_series.split(",")]

            text_description = re.search(r'Text Description: "(.*)"', example).group(1)

            try:
                accuracy_of_trend = re.search(r"Accuracy of trend description: (.*)/5", example).group(1)
                mention_of_seasonality = re.search(r"Mention of seasonality: (.*)/5", example).group(1)
                reference_to_external_factors = re.search(r"Reference to external factors: (.*)/5", example).group(1)
                clarity_of_description = re.search(r"Clarity of description: (.*)/5", example).group(1)
                completeness_of_information = re.search(r"Completeness of information: (.*)/5", example).group(1)
                total_score = re.search(r"Total score: (.*)/25", example).group(1)
                feedback_text = ""

            except Exception as feedback_extraction_error:
                if allow_empty_feedback:
                    feedback_text = ""
                    print(f"[Warning] Feedback missing in example, filled with empty string.")
                else:
                    raise ValueError(f"Feedback extraction failed and allow_empty_feedback=False: {feedback_extraction_error}")


            res.append({
                "time_series": time_series_list,
                "text_description": text_description,
                "accuracy_of_trend": accuracy_of_trend,
                "mention_of_seasonality": mention_of_seasonality,
                "reference_to_external_factors": reference_to_external_factors,
                "clarity_of_description": clarity_of_description,
                "completeness_of_information": completeness_of_information,
                "total_score": total_score,
                "feedback": feedback_text
            })

        except Exception as e:
            print(f"Error parsing example: {e}")

    df = pd.DataFrame(res)
    df.to_json(output_file, orient="records", lines=True)
    print(f"Saved feedback examples to {output_file}")


if __name__ == "__main__":
    timeseries_iterate_prompt_to_json(output_file="./my_feedback_examples.jsonl")
