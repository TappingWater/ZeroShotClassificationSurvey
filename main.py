from methods.svm import run_svm
from methods.nb import run_nb
from methods.nli import run_nli
from methods.embedding import run_sentence_transformer
from utils.report import registry

# Datasets to benchmark with optional per-method limits for speed
DATASETS = {
    "ag_news": {
        "svm_train_limit": None,
        "svm_test_limit": None,
        "nb_train_limit": None,
        "nb_test_limit": None,
        "nli_limit": 100,
        "embedding_test_limit": None,
    },
    "yahoo_answers_topics": {
        "svm_train_limit": None,   # Yahoo is large; cap for quicker runs
        "svm_test_limit": None,
        "nb_train_limit": None,
        "nb_test_limit": None,
        "nli_limit": 100,
        "embedding_test_limit": None,
    },
}

def main():
    print("Initiating Benchmark Suite...")
    
    for dataset, cfg in DATASETS.items():
        print(f"\n=== Dataset: {dataset} ===")
        # 1. Run Supervised Baseline
        try:
            run_svm(
                dataset_name=dataset,
                train_limit=cfg.get("svm_train_limit"),
                test_limit=cfg.get("svm_test_limit"),
            )
        except Exception as e:
            print(f"Error running SVM on {dataset}: {e}")

        # 1b. Run Supervised Naive Bayes Baseline
        try:
            run_nb(
                dataset_name=dataset,
                train_limit=cfg.get("nb_train_limit"),
                test_limit=cfg.get("nb_test_limit"),
            )
        except Exception as e:
            print(f"Error running Naive Bayes on {dataset}: {e}")

        # 2. Run Zero-Shot NLI
        try:
            run_nli(
                dataset_name=dataset,
                limit=cfg.get("nli_limit"),
            )
        except Exception as e:
            print(f"Error running NLI on {dataset}: {e}")

        # 3. Embedding-Based Classifier
        try:
            run_sentence_transformer(
                dataset_name=dataset,
                test_limit=cfg.get("embedding_test_limit"),
            )
        except Exception as e:
            print(f"Error running SentenceTransformer classifier on {dataset}: {e}")

    # 4. Generate Final Report
    print("\nGenerating Consolidated Report...")
    registry.generate_report(title="Text Classification Benchmarks: Supervised vs Zero-Shot")

if __name__ == "__main__":
    main()
