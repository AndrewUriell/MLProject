import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pyhealth.datasets import get_dataloader
from pyhealth.models import RNN
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed

from src.datasets.mortality import (
    load_common_config,
    get_mortality_data_splits,
)


def main():
    set_seed(42)
    config = load_common_config()

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    monitor = config["training"]["monitor"]

    print("Loading mortality dataset and splits...")
    sample_dataset, train_dataset, val_dataset, test_dataset = get_mortality_data_splits()

    print("Creating dataloaders...")
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Building baseline RNN model...")
    model = RNN(dataset=sample_dataset)

    print("Creating trainer...")
    trainer = Trainer(model=model, device="cuda")

    print("Training baseline mortality model...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor=monitor,
        patience=15,
    )

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)

    print("Test metrics:")
    print(test_metrics)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "baseline_mortality_test_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Saved test metrics to: {output_path}")


if __name__ == "__main__":
    main()