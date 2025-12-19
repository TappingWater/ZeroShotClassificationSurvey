from datasets import load_dataset

# Supported datasets and their text/label fields
DATASET_CONFIGS = {
    "ag_news": {
        "hub_name": "ag_news",
        "label_field": "label",
        "text_fields": ["text"],
    },
    "yahoo_answers_topics": {
        "hub_name": "yahoo_answers_topics",
        "label_field": "topic",
        "text_fields": ["question_title", "question_content", "best_answer"],
    },
}


def _compose_text(columns):
    """Join non-empty text fields into one string."""
    parts = [p.strip() for p in columns if p and p.strip()]
    return " ".join(parts)


def _build_split(dataset_split, cfg, limit=None):
    """Return texts and labels for a split, optionally limited."""
    if limit is not None:
        dataset_split = dataset_split.select(range(limit))

    text_columns = [dataset_split[field] for field in cfg["text_fields"]]
    texts = [_compose_text(parts) for parts in zip(*text_columns)]
    labels = dataset_split[cfg["label_field"]]
    return texts, labels


def load_text_classification_dataset(name, train_limit=None, test_limit=None):
    """
    Load and normalize supported text classification datasets.

    Returns (train_texts, train_labels, test_texts, test_labels, label_names)
    with texts merged into a single string per example.
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{name}' not supported. Choose from: {list(DATASET_CONFIGS)}")

    cfg = DATASET_CONFIGS[name]
    dataset = load_dataset(cfg["hub_name"])
    label_names = dataset["train"].features[cfg["label_field"]].names

    train_texts, train_labels = _build_split(dataset["train"], cfg, train_limit)
    test_texts, test_labels = _build_split(dataset["test"], cfg, test_limit)

    return train_texts, train_labels, test_texts, test_labels, label_names
