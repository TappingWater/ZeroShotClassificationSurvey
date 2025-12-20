import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
import glob
import textwrap

def load_results(results_dir):
    data = []
    
    files = glob.glob(str(Path(results_dir) / "*.json"))
    for f in files:
        try:
            res = json.load(open(f))
            res["filename"] = Path(f).stem
            data.append(res)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    return pd.DataFrame(data)

def plot_side_info(df, output_dir):
    """
    Plots Accuracy vs Captions Per Image.
    Include Supervised Upper Bound.
    """
    # Identify Supervised
    sup_row = df[df["model"] == "supervised_upper_bound"]
    sup_acc = sup_row["zsl_gan_acc"].values[0] if not sup_row.empty else 0.0
    
    # Identify Side Info runs (caps_*.json)
    subset = df[df["filename"].str.startswith("caps_")].copy()
    
    if subset.empty:
        print("No Side Info data found.")
        return

    # Map -1 to "All"
    subset["Captions"] = subset["captions_per_image"].apply(lambda x: "All" if x == -1 else str(x))
    
    # Sort: 1, 3, 5, All
    sort_order = ["1", "3", "5", "All"]
    subset["Captions"] = pd.Categorical(subset["Captions"], categories=sort_order, ordered=True)
    subset = subset.sort_values("Captions")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plot Bars
    ax = sns.barplot(data=subset, x="Captions", y="zsl_gan_acc", palette="viridis")
    
    # Add Supervised Line
    if sup_acc > 0:
        plt.axhline(y=sup_acc, color='r', linestyle='--', linewidth=2, label=f"Supervised Upper Bound ({sup_acc:.2%})")
    
    plt.title("ZSL Performance vs. Number of Captions (Side Info)", fontsize=15, fontweight='bold')
    plt.xlabel("Number of Captions per Image", fontsize=13)
    plt.ylabel("ZSL Accuracy (Unseen Classes)", fontsize=13)
    plt.legend()
    
    # Add values on bars
    for i, v in enumerate(subset["zsl_gan_acc"]):
        ax.text(i, v + 0.002, f"{v:.2%}", color='black', ha='center', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    out_path = Path(output_dir) / "plot_side_info.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def plot_data_scaling(df, output_dir):
    """
    Plots Accuracy vs Number of Seen Classes.
    """
    # Identify Supervised
    sup_row = df[df["model"] == "supervised_upper_bound"]
    sup_acc = sup_row["zsl_gan_acc"].values[0] if not sup_row.empty else 0.0

    # Filter for data runs
    subset = df[df["filename"].str.startswith("data_")].copy()
    
    if subset.empty:
        print("No Data Scaling data found.")
        return
        
    # Ensure num_seen_classes is int
    subset["num_seen_classes"] = subset["num_seen_classes"].astype(int)
    subset = subset.sort_values("num_seen_classes")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    sns.lineplot(data=subset, x="num_seen_classes", y="zsl_gan_acc", marker="o", markersize=10, linewidth=3, color="royalblue", label="ZSL (Ours)")
    
    # Add Supervised Line
    if sup_acc > 0:
        plt.axhline(y=sup_acc, color='r', linestyle='--', linewidth=2, label=f"Supervised Upper Bound ({sup_acc:.2%})")

    plt.title("ZSL Performance vs. Number of Seen Classes", fontsize=15, fontweight='bold')
    plt.xlabel("Number of Seen Classes (Training Set)", fontsize=13)
    plt.ylabel("ZSL Accuracy (Unseen Classes)", fontsize=13)
    plt.xticks([20, 50, 75, 110, 150])
    plt.legend()
    
    # Annotate
    for x, y in zip(subset["num_seen_classes"], subset["zsl_gan_acc"]):
        plt.text(x, y + 0.005, f"{y:.2%}", color='black', ha='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    out_path = Path(output_dir) / "plot_data_scaling.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def main():
    # Find latest results dir
    results_root = Path("image_results")
    if not results_root.exists():
        print("No image_results directory found.")
        return
        
    # Sort folders by name (timestamp)
    dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    if not dirs:
        print("No result directories found.")
        return
        
    latest_dir = dirs[-1]
    print(f"Visualizing results from: {latest_dir}")
    
    df = load_results(latest_dir)
    print("Loaded Data:")
    print(df[["filename", "zsl_gan_acc", "num_seen_classes", "captions_per_image"]])
    
    # Create plots output dir inside the results dir
    output_dir = latest_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    plot_side_info(df, output_dir)
    plot_data_scaling(df, output_dir)

if __name__ == "__main__":
    main()
