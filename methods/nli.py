import time
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
from utils.report import registry

# CONFIGURATION
MODEL_NAME = "valhalla/distilbart-mnli-12-3" # Faster
# MODEL_NAME = "facebook/bart-large-mnli"   # More accurate, slower

def format_hypothesis(label):
    """
    The Core NLI Concept:
    Converts a class label (e.g., "Sports") into a logical hypothesis 
    (e.g., "This text is about Sports").
    """
    return f"This text is about {label}."

def run_nli():
    print(f"🔮 Starting Zero-Shot NLI Experiment ({MODEL_NAME})...")

    # 1. Load Data
    dataset = load_dataset("ag_news")
    label_names = dataset['train'].features['label'].names
    print(f"   - Labels Found: {label_names}")

    # NLI is slow on CPU. For testing, limit to 50. 
    # set LIMIT = None to run the full dataset for your final paper.
    LIMIT = 100 
    print(f"   - ⚠️ Running on subset of {LIMIT} samples for speed.")
    
    X_test = dataset['test']['text'][:LIMIT]
    y_test = dataset['test']['label'][:LIMIT]

    # 2. Load NLI Pipeline
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=-1)

    start_time = time.time()
    y_pred = []

    # 3. Inference Loop
    for text in tqdm(X_test, desc="Classifying"):
        
        # The pipeline automatically handles the Premise/Hypothesis logic
        result = classifier(
            text, 
            candidate_labels=label_names,
            hypothesis_template="This text is about {}." 
        )
        
        # Result contains labels sorted by score. Top one is the prediction.
        predicted_label = result['labels'][0]
        
        # Map string back to index (0-3)
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
            "Dataset": "AG News",
            "Train Samples": 0 # Key distinction!
        }
    )

if __name__ == "__main__":
    run_nli()