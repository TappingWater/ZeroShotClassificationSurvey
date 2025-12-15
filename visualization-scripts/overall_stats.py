import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
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
    
    # Normalize Dataset names (case sensitivity) BEFORE sorting/grouping
    # This ensures "ag_news" and "AG News" are treated as the same dataset history
    if 'Dataset' in df.columns:
        df['Dataset'] = df['Dataset'].replace({
            'ag_news': 'AG News',
            'yahoo_answers_topics': 'Yahoo News',
            'yahoo_answers': 'Yahoo News'
        })

    # Sort by timestamp to ensure we get the latest run
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')

    # Group by unique experiment identifiers and take the last one (latest)
    df = df.groupby(['Dataset', 'Method', 'Model'], as_index=False).last()

    # Apply Latency Multiplier for LLM methods
    llm_mask = df['Method'].str.contains('LLM', case=False, na=False)
    df.loc[llm_mask, 'latency_seconds'] = df.loc[llm_mask, 'latency_seconds'] * 10
    
    return df

def create_output_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(project_root, "plots", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def set_style(ax):
    """Applies common style elements to an axis."""
    ax.grid(True, axis='y', linestyle='--', alpha=0.6, color='gray')
    ax.set_axisbelow(True)

def plot_metrics(df, output_dir):
    """
    Plots F1, Accuracy, and Recall side-by-side in one figure (1x3).
    """
    metrics = [
        ('f1_weighted', 'Weighted F1 Score'),
        ('accuracy', 'Accuracy'),
        ('recall_weighted', 'Weighted Recall')
    ]
    
    # Increase figure width to allow spacing
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
    
    # Use a clearer, distinct palette
    sns.set_palette("Set2")
    
    df['Legend_Label'] = df['Method'] + ": " + df['Model']

    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        set_style(ax)
        
        sns.barplot(
            data=df,
            x="Dataset",
            y=metric,
            hue="Legend_Label",
            ax=ax,
            width=0.6,
            edgecolor="black", # Add border for clarity
            linewidth=0.5
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Dataset', fontsize=13)
        ax.set_ylim(0, 1.05)
        
        if i == 0:
            ax.set_ylabel('Score', fontsize=13)
        else:
            ax.set_ylabel('')

        # Remove individual legends
        ax.get_legend().remove()

    # Add a single global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5), title="Model / Method", fontsize=11)
    
    plt.tight_layout()
    # Adjust spacing between plots (wspace) and right margin for legend
    plt.subplots_adjust(wspace=0.15, right=0.88)
    
    output_path = os.path.join(output_dir, "performance_metrics.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved Metrics plot to {output_path}")
    plt.close()

def plot_latency(df, output_dir):
    """
    Plots Latency separately.
    """
    plt.figure(figsize=(12, 7))
    sns.set_palette("Set2")
    
    ax = plt.gca()
    set_style(ax)
    
    df['Legend_Label'] = df['Method'] + ": " + df['Model']

    sns.barplot(
        data=df,
        x="Dataset",
        y="latency_seconds",
        hue="Legend_Label",
        width=0.6,
        edgecolor="black",
        linewidth=0.5
    )
    
    plt.title('Inference Latency (Log Scale) [LLM x10]', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Latency (seconds)', fontsize=13)
    plt.yscale('log')
    plt.xlabel('Dataset', fontsize=13)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Model / Method", fontsize=11)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "latency_comparison.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved Latency plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print("Data loaded. Latest runs per method:")
        print(df[['Dataset', 'Method', 'f1_weighted', 'latency_seconds']])
        
        out_dir = create_output_dir()
        
        plot_metrics(df, out_dir)
        plot_latency(df, out_dir)
        
        print("Done.")
