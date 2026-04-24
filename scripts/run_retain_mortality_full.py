
import sys
import json
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pyhealth.datasets import get_dataloader
from src.models.retain_model import RETAINModel
from pyhealth.trainer import Trainer

from src.datasets.mortality import load_common_config, get_mortality_data_splits


def main():
    config = load_common_config()
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    monitor = config["training"]["monitor"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading mortality dataset and splits...")
    sample_dataset, train_dataset, val_dataset, test_dataset = get_mortality_data_splits()

    print("Creating dataloaders...")
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Building RETAIN model...")
    model = RETAINModel(dataset=sample_dataset, dropout=0.3, embedding_dim=256)



    print("Creating trainer...")
    trainer = Trainer(model=model, device=device)

    print("Training RETAIN mortality model...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor=monitor,
        weight_decay=1e-4,
        patience=5,
    )


    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)

    print("Test metrics:")
    print(test_metrics)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "retain_mortality_test_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Saved test metrics to: {output_path}")


if __name__ == "__main__":
    main()
