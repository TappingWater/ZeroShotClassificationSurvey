import time
import torch
from transformers import pipeline
from tqdm import tqdm
from utils.data import load_text_classification_dataset
from utils.report import registry

# CONFIGURATION
MODEL_NAME = "valhalla/distilbart-mnli-12-3" # Faster
# MODEL_NAME = "facebook/bart-large-mnli"   # More accurate, slower
DEVICE = 0 if torch.cuda.is_available() else -1
DATASET_NAME = "ag_news"  # Switch to "yahoo_answers_topics" to compare
LIMIT = 100               # Set to None for full test set (slower)
BATCH_SIZE = 64            # Increase if GPU memory allows

def format_hypothesis(label):
    """
    The Core NLI Concept:
    Converts a class label (e.g., "Sports") into a logical hypothesis 
    (e.g., "This text is about Sports").
    """
    return f"This text is about {label}."

def run_nli(dataset_name=DATASET_NAME, limit=LIMIT):
    print(f"🔮 Starting Zero-Shot NLI Experiment ({MODEL_NAME}) on {dataset_name}...")

    # 1. Load Data
    _, _, X_test, y_test, label_names = load_text_classification_dataset(
        dataset_name,
        train_limit=0,
        test_limit=limit,
    )
    print(f"   - Labels Found: {label_names}")

    if limit is not None:
        print(f"   - ⚠️ Running on subset of {limit} samples for speed.")

    # 2. Load NLI Pipeline
    device_label = "cuda:0" if DEVICE == 0 else "cpu"
    print(f"   - Using device: {device_label}")
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=DEVICE)

    start_time = time.time()
    y_pred = []

    # 3. Inference Loop (batched for speed)
    for start_idx in tqdm(range(0, len(X_test), BATCH_SIZE), desc="Classifying (batched)"):
        batch_texts = X_test[start_idx:start_idx + BATCH_SIZE]

        results = classifier(
            batch_texts,
            candidate_labels=label_names,
            hypothesis_template="This text is about {}.",
        )

        # Normalize to list for both single and multi input cases
        if not isinstance(results, list):
            results = [results]

        for result in results:
            predicted_label = result["labels"][0]
            y_pred.append(label_names.index(predicted_label))

    end_time = time.time()
    total_time = end_time - start_time

    # 4. Log to Generic Registry
    registry.log_run(
        y_true=y_test,
        y_pred=y_pred,
        time_taken=total_time,
        tags={
            "Method": "Zero-Shot NLI",
            "Model": MODEL_NAME,
            "Dataset": dataset_name,
            "Train Samples": 0 # Key distinction!
        }
    )

if __name__ == "__main__":
    run_nli()
