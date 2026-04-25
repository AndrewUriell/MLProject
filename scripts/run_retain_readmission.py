import sys
import json
from pathlib import Path
 
import torch
 
sys.path.append(str(Path(__file__).resolve().parents[1]))
 
from pyhealth.datasets import get_dataloader
from pyhealth.trainer import Trainer
 
from src.datasets.readmission import load_common_config, get_readmission_data_splits
from src.models.retain_model import RETAINModel
import argparse

 
ABLATION_MODES = {
    "full":     {"use_alpha": True,  "use_beta": True},
    "no_alpha": {"use_alpha": False, "use_beta": True},
    "no_beta":  {"use_alpha": True,  "use_beta": False},
    "empty": {"use_alpha": False,  "use_beta": False},
}
 
ADDITIONAL_FLAGS = {
    "base":  {"focal_loss": False, "bidirectional": False},
    "focal": {"focal_loss": True,  "bidirectional": False},
    "bidir": {"focal_loss": False, "bidirectional": True},
    "full":  {"focal_loss": True,  "bidirectional": True},
}
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(ABLATION_MODES.keys()), default="full")
    parser.add_argument("--variant", choices=list(ADDITIONAL_FLAGS.keys()),       default="full")
    return parser.parse_args()
 
 
def main():
    args = parse_args()
    mode = args.mode
    variant = args.variant
    print(f"Running RETAIN readmission — mode: {mode}")
 
    config = load_common_config()
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    monitor = config["training"]["monitor"]
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
 
    print("Loading readmission dataset and splits...")
    sample_dataset, train_dataset, val_dataset, test_dataset = get_readmission_data_splits()
 
    print("Creating dataloaders...")
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
 
    print(f"Building RETAIN model (mode={mode}),  variant={variant}) ...")
    model = RETAINModel(dataset=sample_dataset, **ABLATION_MODES[mode], **ADDITIONAL_FLAGS[variant],)

 
    trainer = Trainer(model=model, device=device)
 
    print("Training...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor=monitor,
        optimizer_params={"lr": 0.0005},
        patience=15,
    )
 
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    test_metrics["mode"] = mode
 
    print("Test metrics:")
    print(test_metrics)
 
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
 
    output_path = results_dir / f"retain_drug_{mode}_{variant}_test_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
 
    print(f"Saved to: {output_path}")
 
 
if __name__ == "__main__":
    main()