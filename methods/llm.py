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
from openai import OpenAI
from utils.data import load_text_classification_dataset
from utils.report import registry

# CONFIGURATION
MODEL_NAME = "gpt-5-nano"
# Set TEST_LIMIT = None to use the full test set
TEST_LIMIT = 50  # Limiting to 50 for cost/speed during this experiment as recommended for API use
DATASET_NAME = "ag_news"


def run_llm_classification(dataset_name=DATASET_NAME, test_limit=TEST_LIMIT):
    print(f"Starting LLM-Based Zero-Shot Experiment ({MODEL_NAME}) on {dataset_name}...")

    # Load Data
    _, _, X_test, y_test, label_names = load_text_classification_dataset(
        dataset_name,
        train_limit=0,
        test_limit=test_limit,
    )

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print(f"Labels: {label_names}")
    pred_indices = []
    
    start_time = time.time()

    for i, text in enumerate(X_test):
        prompt = (
            f"Classify the following text into one of these categories: {', '.join(label_names)}.\n\n"
            f"Text: {text}\n\n"
            f"Return ONLY the category name."
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful classification assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            content = "ERROR"

        # Naive mapping: find which label name is present in the output
        # If no validation, we just take the first match or -1
        predicted_idx = -1
        for idx, label in enumerate(label_names):
            # Case insensitive check
            if label.lower() in content.lower():
                predicted_idx = idx
                break
        
        # If -1, it means the LLM output something else or hallucinated. 
        # We store -1 to indicate failure to match a valid label.
        pred_indices.append(predicted_idx)
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(X_test)} samples...")

    end_time = time.time()
    total_time = end_time - start_time

    # Log to Generic Registry
    registry.log_run(
        y_true=y_test,
        y_pred=pred_indices,
        time_taken=total_time,
        tags={
            "Method": "LLM Zero-Shot",
            "Model": MODEL_NAME,
            "Dataset": dataset_name,
            "Train Samples": 0,
        },
    )

if __name__ == "__main__":
    run_llm_classification()
