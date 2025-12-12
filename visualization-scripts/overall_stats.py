import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add project root to path to find experiment_results.json
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
results_file = os.path.join(project_root, "experiment_results.json")

def load_data():
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Normalize Dataset names (case sensitivity)
    df['Dataset'] = df['Dataset'].replace({
        'ag_news': 'AG News',
        'yahoo_answers_topics': 'Yahoo News'
    })
    
    return df

def create_output_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(project_root, "plots", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_combined_metrics(df, output_dir):
    """
    Plots F1, Accuracy, and Recall side-by-side in one figure.
    """
    metrics = [
        ('f1_weighted', 'F1 Weighted Score'),
        ('accuracy', 'Accuracy'),
        ('recall_weighted', 'Recall Weighted')
    ]
    
    # Create subplots: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
    sns.set_theme(style="whitegrid")

    # Get the best runs once to ensure consistent coloring across all plots
    # We sort by F1 to pick the "best" run to represent each method/model
    best_runs = df.sort_values('f1_weighted', ascending=False).groupby(['Dataset', 'Method', 'Model']).first().reset_index()
    best_runs['Legend_Label'] = best_runs['Method'] + ": " + best_runs['Model']

    # Get unique labels for the legend to ensure order if needed, 
    # but seaborn handles hue matching automatically if data is same.
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        
        sns.barplot(
            data=best_runs,
            x="Dataset",
            y=metric,
            hue="Legend_Label",
            palette="viridis",
            errorbar=None,
            ax=ax
        )
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylim(0, 1.0)
        
        # Only show y-label for the first plot to save space
        if i == 0:
            ax.set_ylabel('Score', fontsize=12)
        else:
            ax.set_ylabel('')

        # Remove individual legends; we'll add a global one
        ax.get_legend().remove()

    # Add a single global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), title="Model", fontsize=12)
    
    plt.tight_layout()
    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)
    
    output_path = os.path.join(output_dir, "combined_metrics_comparison.png")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved Combined Metrics plot to {output_path}")
    plt.close()

def plot_latency(df, output_dir):
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    best_runs = df.sort_values('f1_weighted', ascending=False).groupby(['Dataset', 'Method', 'Model']).first().reset_index()
    best_runs['Legend_Label'] = best_runs['Method'] + ": " + best_runs['Model']

    chart = sns.barplot(
        data=best_runs,
        x="Dataset",
        y="latency_seconds",
        hue="Legend_Label",
        palette="rocket",
        errorbar=None
    )
    
    plt.title('Inference Latency by Model and Dataset (Log Scale)', fontsize=16)
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.yscale('log')
    plt.xlabel('Dataset', fontsize=12)
    
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., title="Model")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "latency_comparison.png")
    plt.savefig(output_path)
    print(f"Saved Latency plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print("Data loaded. Generating plots...")
        out_dir = create_output_dir()
        
        # Plot Combined Metrics (Side-by-Side)
        plot_combined_metrics(df, out_dir)
        
        # Plot Latency (Separate)
        plot_latency(df, out_dir)
        
        print("Done.")
