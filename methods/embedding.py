import time
import torch
from sentence_transformers import SentenceTransformer, util
from utils.data import load_text_classification_dataset
from utils.report import registry

# CONFIGURATION
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Set TEST_LIMIT = None to use the full test set
TEST_LIMIT = None
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = "ag_news"  # Switch to "yahoo_answers_topics" to compare


def run_sentence_transformer(dataset_name=DATASET_NAME, test_limit=TEST_LIMIT):
    print(f"Starting Embedding-Based Zero-Shot Experiment ({MODEL_NAME}) on {dataset_name}...")

    # Load Data
    _, _, X_test, y_test, label_names = load_text_classification_dataset(
        dataset_name,
        train_limit=0,
        test_limit=test_limit,
    )

    # Build label prompts and encode once
    hypothesis_templates = [f"This text is about {label}." for label in label_names]
    encoder = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"Using device: {DEVICE}")

    print("Encoding label prompts...")
    label_embeddings = encoder.encode(
        hypothesis_templates,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    start_time = time.time()

    # Encode test texts in batches and score via cosine similarity
    print("Encoding test texts and scoring...")
    text_embeddings = encoder.encode(
        X_test,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    scores = util.cos_sim(text_embeddings, label_embeddings)
    pred_indices = scores.argmax(dim=1).tolist()

    end_time = time.time()
    total_time = end_time - start_time

    # Log to Generic Registry
    registry.log_run(
        y_true=y_test,
        y_pred=pred_indices,
        time_taken=total_time,
        tags={
            "Method": "Embedding-Based Zero-Shot",
            "Model": MODEL_NAME,
            "Dataset": dataset_name,
            "Train Samples": 0,
        },
    )


if __name__ == "__main__":
    run_sentence_transformer()
