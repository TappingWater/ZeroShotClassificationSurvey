from methods.svm import run_svm
from methods.nli import run_nli
from utils.report import registry

def main():
    print("Initiating Benchmark Suite...")
    
    # 1. Run Supervised Baseline
    try:
        run_svm()
    except Exception as e:
        print(f"Error running SVM: {e}")

    # 2. Run Zero-Shot NLI
    try:
        run_nli()
    except Exception as e:
        print(f"Error running NLI: {e}")

    # 3. Partner Methods (Placeholders)
    # import embedding_classifier
    # embedding_classifier.run()

    # 4. Generate Final Report
    print("\nGenerating Consolidated Report...")
    registry.generate_report(title="AG News Benchmark: Supervised vs Zero-Shot")

if __name__ == "__main__":
    main()