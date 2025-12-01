# Zero-Shot Classification Survey Experiment

This project benchmarks Zero-Shot Classification (using NLI) against traditional Supervised Learning (using SVM) on the AG News dataset.

## Dependencies

This project relies on the following Python libraries, managed via uv:
- python: >=3.10
- transformers: For NLI models and pipelines.
- torch: PyTorch backend for transformers.
- scikit-learn: For SVM implementation and metric calculations.
- datasets: To download and manage the AG News benchmark.
- pandas: For data manipulation and report generation.
- tqdm: For progress bars during inference.

## Quick Start

1. Install uv
2. Clone the repository
3. Sync dependencies
4. Activate virtual environment
5. Run project

## Project Structure

- main.py: The orchestrator script. It imports methods from the methods/ directory and runs the full benchmark suite.
- methods/svm.py: Implementation of the Supervised LinearSVC baseline (trained on 120k examples).
- methods/nli.py: Implementation of the Zero-Shot NLI classifier (DeBERTa/BART) using hypothesis templates.
- utils/report.py: Shared utility for logging metrics, saving results to JSON, and generating LaTeX tables.
- experiment_results.json: The log file where experiment metrics are persisted. Delete this file to reset the benchmark results.