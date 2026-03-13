# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import pandas as pd
from tqdm import tqdm
import json
import os
import openai


def load_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_dataset_description(dataset_name, json_file="dataset_description_bank.json"):
    """
    Generate a dataset-level description based on metadata in a JSON file.
    """
    try:
        with open(json_file, "r") as file:
            datasets = json.load(file)

        if dataset_name in datasets:
            dataset_info = datasets[dataset_name]
            selected_template = random.choice(dataset_templates)
            return selected_template.format(
                dataset_name=dataset_name,
                frequency=dataset_info["frequency"],
                data_description=dataset_info["data_description"],
                start_date=dataset_info["start_date"],
                end_date=dataset_info["end_date"],
                domain=dataset_info.get("domain", "general")
            )
        else:
            return f"No information found for dataset: {dataset_name}"
    except FileNotFoundError:
        return f"Error: JSON file '{json_file}' not found."
    except KeyError as e:
        return f"Error: Missing key {e} in dataset information."

def compute_variability_summary(std_dev, mean_val):
    """
    Assess the variability level of a time series window based on standard deviation relative to the mean.
    """
    if std_dev / mean_val > 0.5:
        return "high"
    elif std_dev / mean_val > 0.2:
        return "moderate"
    else:
        return "low"

def compute_trend(data):
    """
    Determine the overall trend in a time series window.
    """
    if data[-1] > data[0] * 1.3:
        return "increasing"
    elif data[-1] < data[0] * 0.7:
        return "decreasing"
    else:
        return "fluctuating"

def optimize_text_with_llm(text, model='gpt-4o-2024-05-13', api_key=None):
    """
    Optimize a text description using an LLM (e.g., GPT-4) to improve grammar, clarity, and narrative quality.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    openai.api_key = api_key

    prompt = f"""
    Please rewrite the following description to improve its clarity, grammar, and narrative quality. 
    Keep the technical meaning but make it more professional and natural, as if it were written by an expert data analyst. 
    Avoid redundant phrases. Output only the improved description.

    Text: \"{text}\"
    """

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        optimized_text = response['choices'][0]['message']['content'].strip()
        return optimized_text
    except Exception as e:
        print(f"Error during LLM optimization: {e}")
        return text  # Return original text if LLM call fails

def generate_text_description_for_time_series(file_path, prediction_length, dataset_name,
                                              llm_optimize=False, llm_api_key=None,
                                              dataset_template_file=None, 
                                              description_template_file=None):
    """
    Generate and optionally optimize text descriptions for each time window in a time series dataset.
    """
    # ====== Load external description & templates ======
    dataset_templates = load_json_file(dataset_template_file) if dataset_template_file else []
    description_templates = load_json_file(description_template_file) if description_template_file else []

    print(f"Loaded {len(dataset_templates)} dataset templates")
    print(f"Loaded {len(description_templates)} description templates")

    df = pd.read_csv(file_path)

    for column in df.columns:
        dataset_column_name = f"{column}_dataset_description"
        text_column_name = f"{column}_text_description"
        df[dataset_column_name] = ""
        df[text_column_name] = ""

        for start in tqdm(range(0, len(df), 1), desc=f"Processing column {column}"):
            end = start + prediction_length
            dataset_description = generate_dataset_description(dataset_name,  dataset_templates)

            # If the window exceeds data length, apply the dataset description to remaining rows and break
            if end > len(df):
                df.loc[start:, dataset_column_name] = dataset_description
                df.loc[start:, text_column_name] = dataset_description
                break

            window_data = df[column][start:end].dropna().tolist()

            try:
                window_data = [float(i) for i in window_data]
            except ValueError:
                continue  # Skip non-numeric data windows

            if len(window_data) < prediction_length:
                continue  # Skip incomplete windows

            
            min_value = min(window_data)
            max_value = max(window_data)
            mean_value = round(sum(window_data) / len(window_data), 2)
            std_value = round(pd.Series(window_data).std(), 2)
            peak_steps = [i for i, val in enumerate(window_data) if val == max_value]
            dip_steps = [i for i, val in enumerate(window_data) if val == min_value]
            variability_summary = compute_variability_summary(std_value, mean_value)
            trend = compute_trend(window_data)

            selected_template = random.choice(description_templates)
            text_description = selected_template.format(
                prediction_length=prediction_length,
                min_value=min_value,
                max_value=max_value,
                mean_value=mean_value,
                std_value=std_value,
                peak_steps=peak_steps,
                dip_steps=dip_steps,
                variability_summary=variability_summary + " " + trend,
                total_steps=len(df[column])
            )


            if llm_optimize:
                optimized_text = optimize_text_with_llm(
                    text=text_description,
                    model='gpt-4o-2024-05-13',
                    api_key=llm_api_key
                )
                print(f"\nOriginal: {text_description}\nOptimized: {optimized_text}\n")
                text_description = optimized_text 

            df.at[start, dataset_column_name] = dataset_description
            df.at[start, text_column_name] = text_description

    output_file_path = file_path.replace(".csv", "_with_descriptions.csv")
    df.to_csv(output_file_path, index=False)
    print(f"Dataset and text descriptions saved to {output_file_path}")
