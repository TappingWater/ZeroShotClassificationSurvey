import sys
import os

# Add project root to sys.path to allow running this script directly
# This allows 'import utils...' to work when running 'python methods/llm.py'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import time
import json
import concurrent.futures
from tqdm import tqdm
import google.generativeai as genai
from openai import OpenAI
from utils.data import load_text_classification_dataset
from utils.report import registry

# CONFIGURATION DEFAULTS
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_DATASET = "ag_news"
DEFAULT_PROVIDER = "openai"
TEST_LIMIT = None  # Process full dataset by default
MAX_WORKERS = 10   # Number of parallel requests


def get_llm_response(provider, model_name, prompt, client=None):
    """
    Get response from the specified LLM provider.
    """
    if provider == "openai":
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful classification assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return "ERROR"

    elif provider == "google":
        try:
            # Gemini client logic
            model = client.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "ERROR"
            
    else:
        raise ValueError(f"Unknown provider: {provider}")


def classify_single_text(text, label_names, provider, model_name, client):
    """
    Helper function to classify a single text sample.
    Returns the predicted index.
    """
    prompt = (
        f"Classify the following text into one of these categories: {', '.join(label_names)}.\n\n"
        f"Text: {text}\n\n"
        f"Return ONLY the category name."
    )

    content = get_llm_response(provider, model_name, prompt, client)

    # Naive mapping: find which label name is present in the output
    predicted_idx = -1
    for idx, label in enumerate(label_names):
        # Case insensitive check
        if label.lower() in content.lower():
            predicted_idx = idx
            break
            
    return predicted_idx


def run_llm_classification(dataset_name=DEFAULT_DATASET, model_name=DEFAULT_MODEL, provider=DEFAULT_PROVIDER, test_limit=TEST_LIMIT):
    print(f"Starting LLM-Based Zero-Shot Experiment ({model_name} | {provider}) on {dataset_name}...")
    
    if test_limit is None:
        print("Model configured to run on FULL test set.")
    else:
        print(f"Model configured to run on {test_limit} samples.")

    # Load Data
    _, _, X_test, y_test, label_names = load_text_classification_dataset(
        dataset_name,
        train_limit=0,
        test_limit=test_limit,
    )

    # Initialize Client
    client = None
    if provider == "openai":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif provider == "google":
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        client = genai
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    print(f"Labels: {label_names}")
    print(f"Processing {len(X_test)} samples with {MAX_WORKERS} workers...")
    
    start_time = time.time()
    
    # Pre-bind arguments for the helper function
    def task(text):
        return classify_single_text(text, label_names, provider, model_name, client)

    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pred_indices = list(tqdm(executor.map(task, X_test), total=len(X_test)))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Completed in {total_time:.2f} seconds.")

    # --- SAVE PREDICTIONS TO JSON ---
    output_dir = os.path.join("runs", "llm", model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}.json")

    results_data = {
        "dataset": dataset_name,
        "model": model_name,
        "provider": provider,
        "label_names": label_names,
        "y_true": y_test,
        "y_pred": pred_indices,
        "time_taken": total_time
    }

    try:
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Failed to save predictions: {e}")
    # --------------------------------

    # Log to Generic Registry
    registry.log_run(
        y_true=y_test,
        y_pred=pred_indices,
        time_taken=total_time,
        tags={
            "Method": "LLM Zero-Shot",
            "Model": model_name,
            "Dataset": dataset_name,
            "Provider": provider,
            "Train Samples": 0,
        },
    )

if __name__ == "__main__":
    # Example usage - can be modified here for testing different configs
    # run_llm_classification(dataset_name="ag_news", model_name="gpt-5-nano", provider="openai")
    
    # Running Yahoo News (Full Test Set)
    run_llm_classification(dataset_name="yahoo_answers_topics", model_name="gpt-5-nano", provider="openai", test_limit=None)
