import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from datetime import datetime

LOG_FILE = "experiment_results.json"

class ExperimentRegistry:
    def __init__(self, log_file=LOG_FILE):
        self.log_file = log_file
        # Initialize file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)

    def log_run(self, y_true, y_pred, time_taken, tags=None):
        """
        Generic method to log any classification run.
        """
        if tags is None:
            tags = {}

        # Calculate standard metrics
        metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_weighted": round(f1_score(y_true, y_pred, average='weighted'), 4),
            "recall_weighted": round(recall_score(y_true, y_pred, average='weighted'), 4),
            "latency_seconds": round(time_taken, 2)
        }

        # Merge user tags with calculated metrics
        entry = {**tags, **metrics}

        # Load existing data, append new entry, and save
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        data.append(entry)

        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"[Log] Run logged: {tags.get('Method', 'Unknown')} | F1: {metrics['f1_weighted']}")

    def generate_report(self, title="Experiment Summary"):
        """
        Reads the JSON log and prints a Pandas DataFrame and LaTeX table.
        """
        if not os.path.exists(self.log_file):
            print("No results found.")
            return

        with open(self.log_file, 'r') as f:
            data = json.load(f)

        if not data:
            print("Log file is empty.")
            return

        df = pd.DataFrame(data)
        
        # Reorder columns for readability
        cols = ['Method', 'Model', 'Dataset', 'Train Samples', 'f1_weighted', 'accuracy', 'recall_weighted', 'latency_seconds']
        # Only select columns that exist in the data
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        print("\n" + "="*50)
        print(title.upper())
        print("="*50)
        print(df.to_string(index=False))

        print("\n" + "="*50)
        print("LATEX TABLE EXPORT")
        print("="*50)
        print(df.to_latex(index=False, float_format="%.4f", caption=title, label="tab:results"))

# Singleton instance for easy import
registry = ExperimentRegistry()

if __name__ == "__main__":
    registry.generate_report()
