import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils.data import load_text_classification_dataset
from utils.report import registry

# CONFIGURATION
DATASET_NAME = "ag_news"  # Switch to "yahoo_answers_topics" to compare
TRAIN_LIMIT = 120_000     # Set to None for full dataset (Yahoo is large; cap by default)
TEST_LIMIT = 10_000       # Set to None for full test set
MAX_FEATURES = 20000
ALPHA = 0.1


def run_nb(
    dataset_name=DATASET_NAME,
    train_limit=TRAIN_LIMIT,
    test_limit=TEST_LIMIT,
    max_features=MAX_FEATURES,
    alpha=ALPHA,
):
    print(f"Starting Supervised Naive Bayes Experiment ({dataset_name})...")

    # Load Data (with optional limits)
    X_train, y_train, X_test, y_test, _ = load_text_classification_dataset(
        dataset_name,
        train_limit=train_limit,
        test_limit=test_limit,
    )

    start_time = time.time()

    # Vectorization (TF-IDF)
    print("   - Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Multinomial Naive Bayes
    print("   - Training MultinomialNB...")
    nb = MultinomialNB(alpha=alpha)
    nb.fit(X_train_vec, y_train)

    # Inference
    print("   - Predicting...")
    y_pred = nb.predict(X_test_vec)

    end_time = time.time()
    total_time = end_time - start_time

    # Log to Generic Registry
    registry.log_run(
        y_true=y_test,
        y_pred=y_pred,
        time_taken=total_time,
        tags={
            "Method": "Supervised",
            "Model": "MultinomialNB",
            "Dataset": dataset_name,
            "Train Samples": len(X_train),
        },
    )


if __name__ == "__main__":
    run_nb()
