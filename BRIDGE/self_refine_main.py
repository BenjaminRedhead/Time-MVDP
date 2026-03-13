# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import json
import pandas as pd
from typing import List, Dict, Any

from llm_agents import Agent, ChatLLM, PythonREPLTool, HackerNewsSearchTool, SerpAPITool
from self_refine.task_init import TimeSeriesTaskInit
from self_refine.task_iterate import TimeSeriesTaskIterate
from self_refine.feedback import TimeSeriesFeedback
from ts_to_text import generate_text_description_for_time_series
from self_refine.prompt_building import timeseries_iterate_prompt_to_json
from template_extractor import TemplateExtractor

# ========== CONFIGURATION ==========
class Config:
    openai_api_key = "xxx"
    # If you want to connect to the real world, please set up your own specific search engine and API
    # e.g.***_api_key = "xxx"
    # ***_search_engine = "xxx"

# ========== UTILITIES ==========
def save_to_csv(data, filename):
    print(">>> DEBUG: Saving data")
    print(">>> Type:", type(data))
    print(">>> Preview:", str(data)[:500])

    try:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
            df.to_csv(filename, index=False)
            print(f"DataFrame (dict) saved to {filename}")

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                print(f"DataFrame (list of dicts) saved to {filename}")
            else:
                df = pd.DataFrame({"values": data})
                df.to_csv(filename, index=False)
                print(f"DataFrame (list of values) saved to {filename}")

        elif isinstance(data, str):
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(data)
                print(f"String saved to {filename}")

        else:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(str(data))
                print(f"Unknown data type. Raw text saved to {filename}")

    except Exception as e:
        print(f"Error saving data to {filename}: {e}")

# ========== DESCRIPTION GENERATOR ==========
class DescriptionGenerator:
    """
    DescriptionGenerator now wraps around ts_to_text.generate_text_description_for_time_series()
    """
    def __init__(self, dataset_name, prediction_length=168, json_file="dataset_description_bank.json",
                 llm_optimize=False, llm_api_key=None, dataset_template_file=None, description_template_file=None):
        
        self.dataset_name = dataset_name
        self.prediction_length = prediction_length
        self.json_file = json_file
        self.llm_optimize = llm_optimize
        self.llm_api_key = llm_api_key
        self.dataset_template_file = dataset_template_file
        self.description_template_file = description_template_file

    def generate_description(self, file_path):
        print("\n>>> Calling ts_to_text.generate_text_description_for_time_series()...")

        generate_text_description_for_time_series(
            file_path=file_path,
            prediction_length=self.prediction_length,
            dataset_name=self.dataset_name,
            json_file=self.json_file,
            llm_optimize=self.llm_optimize,
            llm_api_key=self.llm_api_key,
            dataset_template_file=self.dataset_template_file,
            description_template_file=self.description_template_file
        )

        print(f">>> Descriptions generated and saved to {file_path.replace('.csv', '_with_descriptions.csv')}")


# ========== MAIN LOGIC ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time series data processing pipeline')


    parser.add_argument('--openai_key', type=str, default=None, help='OpenAI API key')

    # LLM settings
    parser.add_argument('--openai_model', type=str, default='gpt-4o-2024-05-13', help='OpenAI model name')
    parser.add_argument('--llm_optimize', action='store_true', help='Use LLM to optimize text descriptions')

    # Time series files and descriptions
    parser.add_argument('--ts_file', type=str, help='CSV file containing time series data')
    parser.add_argument('--dataset_name', type=str, required=False, help='Name/ID of the dataset for description templates')
    parser.add_argument('--prediction_length', type=int, default=168, help='Prediction length for time series windows')
    parser.add_argument('--dataset_template_file', type=str, help='Path to dataset template JSON file')
    parser.add_argument('--description_template_file', type=str, help='Path to description template JSON file')
    parser.add_argument('--json_file', type=str, help='Dataset description JSON file, default=dataset_description_bank.json')

    # Output files
    parser.add_argument('--output_file', type=str, help='Output file for descriptions or refinement results')
    parser.add_argument('--output_template_file', type=str, default="templates/description_template.json", help='Path to save extracted templates')

    # Candidate collection / extraction
    parser.add_argument('--collect_candidate', action='store_true', help='Collect text candidates from the web')
    parser.add_argument('--extract_template', action='store_true', help='Extract templates from documents')
    parser.add_argument('--template_input_file', type=str, help='Input file path for template extraction')

    # Feedback & Evaluation
    parser.add_argument('--feedback_example_file', type=str, default=None, help='Path to the feedback examples JSON file for initializing TimeSeriesFeedback')

    # Self-Refinement (legacy or single-agent)
    parser.add_argument('--refine', action='store_true', help='Run the multi-agent self-refinement process')
    parser.add_argument('--refine_iterations', type=int, default=10, help='Max attempts for legacy refinement iterations')
    parser.add_argument('--tests', type=int, default=5, help='Number of refinement tests (legacy or multi-run)')

    # Multi-Agent Refiner specific hyperparams
    parser.add_argument('--team_iterations', type=int, default=3, help='Max iterations per team (micro/macro) before convergence or termination')
    parser.add_argument('--global_iterations', type=int, default=2, help='Max global refinement rounds where both teams collaborate and manager gives final decision')

    # Predictor (optional)
    parser.add_argument('--predictor_method', type=str, choices=['llmtime', 'promptcast'], help='Prediction method for generating new time series during refinement')

    args = parser.parse_args()

    # === ENVIRONMENT VARIABLES ===
    if args.openai_key:
        os.environ['OPENAI_API_KEY'] = args.openai_key
        print(f">>> OpenAI API Key set.")

    # You search engine here

    # === AGENT CANDIDATE COLLECTION ===
    if args.collect_candidate:
        print("\n=== Collecting Candidates ===")
        agent = Agent(
            llm=ChatLLM(api_key=args.openai_key),
            tools=[PythonREPLTool(), SerpAPITool(), HackerNewsSearchTool()]
        )

        query_template = """
        Collect as many as you can the essay/report/text/paper that describes time series data (For example: line graph) on any domains (e.g. medical, finance, industry etc.). 
        Return the content and link instead of only title.
        """

        result = agent.run(query_template)
        save_to_csv(result, "output.csv")
        print(f"\nFinal collected results saved to output.csv")

    # === FEEDBACK EXAMPLE HANDLING ===
    if args.feedback_example_file:
        feedback_example_file = args.feedback_example_file
    else:
        # auto fallback and generation
        auto_feedback_file = "./feedback.jsonl"
        if not os.path.exists(auto_feedback_file):
            print(f"\n>>> No feedback example file provided. Auto-generating at {auto_feedback_file}")
            timeseries_iterate_prompt_to_json(output_file=auto_feedback_file)
    
        feedback_example_file = auto_feedback_file

    
    # ========== TEMPLATE EXTRACTOR ==========
    if args.extract_template:
        print("\n=== Extracting Templates ===")

        if not args.template_input_file:
            raise ValueError("--template_input_file is required for extracting templates")

        # Load LLM + FeedbackTool
        llm = ChatLLM(model=args.openai_model, api_key=args.openai_key, temperature=0.0)
        feedback_tool = TimeSeriesFeedback(
            model=args.openai_model,
            input_instance=feedback_example_file
        )

        extractor = TemplateExtractor(
            llm=llm,
            feedback_tool=feedback_tool,
            output_template_file=args.output_template_file
        )

        if args.template_input_file.endswith("multi_agent_refinement_result.json"):
            print("\n>>> Extracting template from multi-agent refinement result...")
            extractor.extract_templates_from_refinement_result(refinement_json_path=args.template_input_file)

        else:
            print("\n>>> Extracting templates from raw document...")
            extractor.extract_templates(input_path=args.template_input_file)

        print("\nTemplate extraction complete.")




    # === TIME SERIES TO TEXT ===
    if args.ts_to_text:
        print("\n=== Generating Descriptions from Time Series ===")

        if not args.dataset_name:
            raise ValueError("--dataset_name is required for ts_to_text conversion")

        generator = DescriptionGenerator(
            dataset_name=args.dataset_name,
            prediction_length=args.prediction_length,
            llm_optimize=args.llm_optimize,
            llm_api_key=args.openai_key,
            dataset_template_file=args.dataset_template_file,
            description_template_file=args.description_template_file
        )

        generator.generate_description(file_path=args.ts_file)

    # === SELF-REFINEMENT LOOP (MULTI-AGENT VERSION) ===
    if args.refine:
        print("\n=== Running Multi-Agent Self-Refinement ===")

        # === STEP 1: Validate Inputs ===
        if not args.ts_file:
            raise ValueError("You must provide --ts_file containing the time series data.")
        if not args.dataset_name:
            raise ValueError("You must provide --dataset_name so we can load its initial description.")

        # === STEP 2: Load the Time Series (actual_series) ===
        print(f"\n>>> Loading time series from {args.ts_file}...")
        ts_df = pd.read_csv(args.ts_file)
        
        # Example: Assume first column is timestamp, second column is the values
        actual_series = ts_df.iloc[:, 1].tolist()
        
        print(f"Loaded actual_series (length {len(actual_series)}). Preview: {actual_series[:10]}")

        # === STEP 3: Load the Text Description (initial_text) ===
        print(f"\n>>> Loading text description for dataset {args.dataset_name}...")
        description_file = args.json_file or "dataset_description_bank.json"
        
        if not os.path.exists(description_file):
            raise FileNotFoundError(f"{description_file} does not exist. Please run ts_to_text generation first.")
        
        with open(description_file, "r", encoding="utf-8") as f:
            description_data = json.load(f)

        # Try to match the dataset_name
        dataset_entry = next((entry for entry in description_data if entry.get("dataset_name") == args.dataset_name), None)
        
        if not dataset_entry:
            raise ValueError(f"No description found for dataset_name '{args.dataset_name}' in {description_file}.")
        
        initial_text = dataset_entry.get("description") or dataset_entry.get("future_description")

        print(f"Loaded initial_text for {args.dataset_name}:\n{initial_text}")

        # === STEP 4: Initialize LLM + FeedbackTool ===
        llm = ChatLLM(model=args.openai_model, api_key=args.openai_key, temperature=0.0)
        
        feedback_tool = TimeSeriesFeedback(
            model=args.openai_model,
            input_instance=args.feedback_example_file or None
        )

        # === STEP 5: Initialize Multi-Agent Refiner ===
        from multi_agent_refiner import MultiAgentRefiner
        
        refiner = MultiAgentRefiner(
            llm=llm,
            feedback_tool=feedback_tool,
            predictor=None,  # Optional predictor can be passed if generating new ts predictions
            actual_series=actual_series,
            team_max_iterations=args.team_iterations,
            global_max_iterations=args.global_iterations
        )

        # === STEP 6: Run Multi-Agent Refinement ===
        final_text, refinement_logs = refiner.refine(initial_text)

        # === STEP 7: Save Final Result ===
        output_file = args.output_file or f"{args.dataset_name}_multi_agent_refinement_result.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(refinement_logs, f, indent=4)

        print(f"\n Multi-Agent Refinement Complete. Results saved to {output_file}")
        print(f"\nFinal Refined Text:\n{final_text}")
