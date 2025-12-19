import sys
import os

# Add project root to sys.path to allow running this script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

import time
import json
import concurrent.futures
from concurrent.futures import as_completed
from tqdm import tqdm
import google.generativeai as genai
from openai import OpenAI
from utils.data import load_text_classification_dataset
from utils.report import registry

# CONFIGURATION DEFAULTS
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_DATASET = "ag_news"
DEFAULT_PROVIDER = "openai"
DEFAULT_LIMIT = None # Limit for NEW samples to process this run
MAX_WORKERS = 10
SAVE_INTERVAL = 20 # Save every N samples

def get_llm_response(provider, model_name, prompt, client=None):
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
            model = client.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "ERROR"
    else:
        raise ValueError(f"Unknown provider: {provider}")


def classify_single_text(text, label_names, provider, model_name, client):
    prompt = (
        f"Classify the following text into one of these categories: {', '.join(label_names)}.\n\n"
        f"Text: {text}\n\n"
        f"Return ONLY the category name."
    )

    content = get_llm_response(provider, model_name, prompt, client)

    predicted_idx = -1
    for idx, label in enumerate(label_names):
        if label.lower() in content.lower():
            predicted_idx = idx
            break
            
    return predicted_idx


def load_checkpoint(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Checkpoint file corrupted or empty. Starting fresh.")
            return None
    return None

def save_checkpoint(filepath, data):
    # Atomic write to avoid corruption
    temp_path = filepath + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(temp_path, filepath)


def run_llm_classification(dataset_name=DEFAULT_DATASET, model_name=DEFAULT_MODEL, provider=DEFAULT_PROVIDER, limit=DEFAULT_LIMIT):
    print(f"Starting LLM-Based Zero-Shot Experiment ({model_name} | {provider}) on {dataset_name}...")
    
    # 1. Setup paths
    output_dir = os.path.join("runs", "llm", model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}.json")
    
    # 2. Load Dataset (Full)
    # We always load full dataset to ensure index alignment
    _, _, X_test, y_test, label_names = load_text_classification_dataset(
        dataset_name,
        train_limit=0,
        test_limit=None, 
    )
    print(f"Labels: {label_names}")
    print(f"Total Test Samples: {len(X_test)}")

    # 3. Load Checkpoint / Initialize State
    checkpoint = load_checkpoint(output_file)
    if checkpoint:
        print(f"Loaded checkpoint from {output_file}")
        # 'predictions' keys are strings in JSON, we'll keep them as strings or convert to int for logic
        # format: {"0": 1, "1": 2}
        existing_predictions = checkpoint.get("predictions", {})
        # Ensure y_true matches? We assume dataset is static.
    else:
        print("No checkpoint found. Starting fresh.")
        existing_predictions = {}
    
    print(f"Already classified: {len(existing_predictions)}")

    # 4. Identify Pending Work
    # Indices that are NOT in existing_predictions
    all_indices = range(len(X_test))
    pending_indices = [i for i in all_indices if str(i) not in existing_predictions]
    
    if not pending_indices:
        print("All samples classified! Calculating metrics...")
        limit = 0 # Skip processing
    
    if limit is not None:
        print(f"Limit applied: processing up to {limit} NEW samples.")
        pending_indices = pending_indices[:limit]
    
    print(f"Samples to process in this run: {len(pending_indices)}")

    # 5. Initialize Client
    client = None
    if pending_indices:
        if provider == "openai":
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif provider == "google":
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            client = genai
    
    # 6. Run Processing (if any)
    predictions_map = existing_predictions.copy() # Local copy to update
    
    start_time = time.time()
    
    if len(pending_indices) > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map future -> original index
            future_to_idx = {
                executor.submit(classify_single_text, X_test[idx], label_names, provider, model_name, client): idx
                for idx in pending_indices
            }
            
            completed_count = 0
            
            # Process as they complete
            for future in tqdm(as_completed(future_to_idx), total=len(pending_indices), desc="Classifying"):
                idx = future_to_idx[future]
                try:
                    pred_idx = future.result()
                    predictions_map[str(idx)] = pred_idx
                    completed_count += 1
                    
                    # Periodic Save
                    if completed_count % SAVE_INTERVAL == 0:
                        save_data = {
                            "dataset": dataset_name,
                            "model": model_name,
                            "provider": provider,
                            "label_names": label_names,
                            "y_true": y_test, # Save full truth for self-containment
                            "predictions": predictions_map,
                            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        save_checkpoint(output_file, save_data)
                        
                except Exception as e:
                    print(f"Exception for sample {idx}: {e}")

    total_time = time.time() - start_time
    print(f"Run finished. Total predictions stored: {len(predictions_map)}/{len(X_test)}")

    # 7. Final Save
    save_data = {
        "dataset": dataset_name,
        "model": model_name,
        "provider": provider,
        "label_names": label_names,
        "y_true": y_test,
        "predictions": predictions_map,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_checkpoint(output_file, save_data)
    print(f"Checkpoint saved to {output_file}")

    # 8. Calculate Metrics
    # Only calculate if we have predictions
    if len(predictions_map) > 0:
        # Reconstruct list for metrics (fill missing with -1)
        full_preds = []
        full_true = []
        valid_count = 0
        
        # We only evaluate on what we have? Or on the full set?
        # Standard: Evaluate on full set, missing = error? 
        # Or usually evaluate on 'so far'? 
        # Let's evaluate on intersection to be useful during progress.
        
        eval_preds = []
        eval_true = []
        
        for i in range(len(X_test)):
            if str(i) in predictions_map:
                eval_preds.append(predictions_map[str(i)])
                eval_true.append(y_test[i])
        
        if len(eval_preds) > 0:
            # Log to Generic Registry (Partial or Full)
            registry.log_run(
                y_true=eval_true,
                y_pred=eval_preds,
                time_taken=total_time, # Note: this is time for THIS run, not cumulative. 
                tags={
                    "Method": "LLM Zero-Shot",
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Provider": provider,
                    "Train Samples": 0,
                    "Evaluated Samples": len(eval_preds)
                },
            )
        else:
            print("No predictions to evaluate.")

if __name__ == "__main__":
    # Run configuration
    
    # AG NEWS (Small, fast check)
    # run_llm_classification(dataset_name="ag_news", model_name="gpt-5-nano", provider="openai")
    
    # YAHOO NEWS (Large, use limit if needed)
    # Example: run 1000 samples, then stop. Run again to resume.
    run_llm_classification(
        dataset_name="yahoo_answers_topics",
        model_name="gpt-5-nano", 
        provider="openai",
        limit=None # Set to an integer to test batching, e.g., 50
    )
