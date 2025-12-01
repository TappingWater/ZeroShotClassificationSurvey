import time
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from utils.report import registry  # Import our generic reporter

def run_svm():
    print("🚀 Starting Supervised SVM Experiment...")
    
    # 1. Load Data
    dataset = load_dataset("ag_news")
    X_train = dataset['train']['text']
    y_train = dataset['train']['label']
    
    # Using full test set
    X_test = dataset['test']['text']
    y_test = dataset['test']['label']

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
            "Dataset": "AG News",
            "Train Samples": 120000
        }
    )

if __name__ == "__main__":
    run_svm()