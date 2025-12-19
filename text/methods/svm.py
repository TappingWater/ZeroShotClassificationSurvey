import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from utils.data import load_text_classification_dataset
from utils.report import registry  # Import our generic reporter

# CONFIGURATION
DATASET_NAME = "ag_news"  # Switch to "yahoo_answers_topics" to compare
TRAIN_LIMIT = 120_000     # Set to None for full dataset (Yahoo is large; cap by default)
TEST_LIMIT = 10_000       # Set to None for full test set


def run_svm(dataset_name=DATASET_NAME, train_limit=TRAIN_LIMIT, test_limit=TEST_LIMIT):
    print(f"🚀 Starting Supervised SVM Experiment ({dataset_name})...")
    
    # 1. Load Data (with optional limits)
    X_train, y_train, X_test, y_test, _ = load_text_classification_dataset(
        dataset_name,
        train_limit=train_limit,
        test_limit=test_limit,
    )
    
    start_time = time.time()

    # 2. Vectorization (TF-IDF)
    print("   - Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 3. Train SVM
    print("   - Training LinearSVC...")
    svm = LinearSVC(dual='auto', random_state=42)
    svm.fit(X_train_vec, y_train)

    # 4. Inference
    print("   - Predicting...")
    y_pred = svm.predict(X_test_vec)
    
    end_time = time.time()
    total_time = end_time - start_time

    # 5. Log to Generic Registry
    registry.log_run(
        y_true=y_test,
        y_pred=y_pred,
        time_taken=total_time,
        tags={
            "Method": "Supervised",
            "Model": "LinearSVC",
            "Dataset": dataset_name,
            "Train Samples": len(X_train)
        }
    )

if __name__ == "__main__":
    run_svm()
