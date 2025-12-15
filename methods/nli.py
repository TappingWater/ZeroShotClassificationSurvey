import sys
import os

# Add project root to sys.path to allow running this script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import torch
from transformers import pipeline
from tqdm import tqdm
from utils.data import load_text_classification_dataset
from utils.report import registry

# CONFIGURATION DEFAULTS
DEFAULT_MODEL = "valhalla/distilbart-mnli-12-3"
DEFAULT_BATCH_SIZE = 64

# Device selection: CUDA -> MPS -> CPU
if torch.cuda.is_available():
    DEVICE = 0
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = -1

def run_nli(dataset_name="yahoo_answers_topics", model_name=DEFAULT_MODEL, limit=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Run Zero-Shot NLI Classification.
    
    Args:
        dataset_name (str): Name of the dataset to load (e.g., 'ag_news', 'yahoo_answers_topics').
        model_name (str): HuggingFace model name for zero-shot classification.
        limit (int, optional): Number of test samples to use. None for all.
        batch_size (int): Batch size for inference.
    """
    print(f"🔮 Starting Zero-Shot NLI Experiment ({model_name}) on {dataset_name}...")
    if limit is None:
        print("   - 🚀 Running on FULL test set (no limit). This may take a while.")
    else:
        print(f"   - ⚠️  Running on subset of {limit} samples.")

    # 1. Load Data
    _, _, X_test, y_test, label_names = load_text_classification_dataset(
        dataset_name,
        train_limit=0,
        test_limit=limit,
    )
    print(f"   - Labels Found: {label_names}")
    print(f"   - Test Set Size: {len(X_test)}")

    # 2. Load NLI Pipeline
    if isinstance(DEVICE, int) and DEVICE != -1:
        device_name = f"cuda:{DEVICE}"
    else:
        device_name = str(DEVICE) if DEVICE != -1 else "cpu"
    print(f"   - Using device: {device_name}")
    
    try:
        classifier = pipeline("zero-shot-classification", model=model_name, device=DEVICE)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return

    start_time = time.time()
    y_pred = []

    # 3. Inference Loop (batched for speed)
    print(f"   - Inference Batch Size: {batch_size}")
    
    # We loop manually to handle potentially large datasets gracefully with tqdm
    for start_idx in tqdm(range(0, len(X_test), batch_size), desc="Classifying"):
        batch_texts = X_test[start_idx:start_idx + batch_size]

        try:
            results = classifier(
                batch_texts,
                candidate_labels=label_names,
                hypothesis_template="This text is about {}.",
            )
        except Exception as e:
            print(f"Error during batch inference at index {start_idx}: {e}")
            # Failsafe: pad predictions with -1 or skip? 
            # Appending -1 for each failed item to maintain length alignment
            y_pred.extend([-1] * len(batch_texts))
            continue

        # Normalize to list for both single and multi input cases
        if not isinstance(results, list):
            results = [results]

        for result in results:
            # result['labels'] tells us the sorted labels by probability
            # result['scores'] tells us the scores
            # The top one is the prediction
            top_label = result["labels"][0]
            
            if top_label in label_names:
                y_pred.append(label_names.index(top_label))
            else:
                # Should not happen given candidate_labels, but good to be safe
                y_pred.append(-1)

    end_time = time.time()
    total_time = end_time - start_time

    # 4. Log to Generic Registry
    registry.log_run(
        y_true=y_test,
        y_pred=y_pred,
        time_taken=total_time,
        tags={
            "Method": "Zero-Shot NLI",
            "Model": model_name,
            "Dataset": dataset_name,
            "Train Samples": 0
        }
    )

if __name__ == "__main__":
    # Default execution: Run on Yahoo Answers with limit=50 for verification
    run_nli(dataset_name="yahoo_answers_topics", limit=None)
