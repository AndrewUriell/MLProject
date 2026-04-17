import sys
import json
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pyhealth.datasets import get_dataloader
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer

from src.datasets.drug_recommendation import (
    load_common_config,
    get_drug_recommendation_data_splits,
)
from src.utils.metrics import precision_recall_at_k


def main():
    config = load_common_config()

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]

    print("Loading drug recommendation dataset and splits...")
    sample_dataset, train_dataset, val_dataset, test_dataset = (
        get_drug_recommendation_data_splits()
    )

    print("Creating dataloaders...")
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Building baseline Transformer model...")
    model = Transformer(dataset=sample_dataset)

    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        device="cuda",
        metrics=["jaccard_samples", "f1_samples", "pr_auc_samples"],
    )

    print("Training baseline drug recommendation model...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="pr_auc_samples",
    )

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)

    # Custom top-k metrics
    y_true, y_prob, _ = trainer.inference(test_loader)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    p10, r10 = precision_recall_at_k(y_true, y_prob, 10)
    p20, r20 = precision_recall_at_k(y_true, y_prob, 20)

    test_metrics["precision_at_10"] = p10
    test_metrics["recall_at_10"] = r10
    test_metrics["precision_at_20"] = p20
    test_metrics["recall_at_20"] = r20

    print("Test metrics:")
    print(test_metrics)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "baseline_drug_test_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Saved test metrics to: {output_path}")


if __name__ == "__main__":
    main()