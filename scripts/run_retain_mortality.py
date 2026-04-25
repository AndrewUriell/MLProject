
import sys
import json
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pyhealth.datasets import get_dataloader
from src.models.retain_model import RETAINModel
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed
import argparse

from src.datasets.mortality import load_common_config, get_mortality_data_splits

#Flags for abalation
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
    parser.add_argument("--variant", choices=list(ADDITIONAL_FLAGS.keys()),       default="false")
    parser.add_argument("--pos_weight", action="store_true")
    parser.add_argument("--pos_weight_scale", type=float, default=1.0)
    return parser.parse_args()

def compute_pos_weight(train_dataset, label_key):
    """pos_weight = N_negative / N_positive, computed on the training split only."""
    num_pos, num_neg = 0, 0
    for sample in train_dataset:
        label = sample[label_key]
        # handle tensor or scalar
        val = label.item() if hasattr(label, "item") else label
        if val == 1:
            num_pos += 1
        else:
            num_neg += 1
    ratio = num_neg / max(num_pos, 1)
    print(f"Positives: {num_pos}, Negatives: {num_neg}, pos_weight: {ratio:.3f}")
    return torch.tensor([ratio], dtype=torch.float32)

def main():
    set_seed(42)
    args = parse_args()
    mode = args.mode
    variant = args.variant
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

    pos_weight_tensor = None
    if args.pos_weight:
        pw = compute_pos_weight(train_dataset, label_key="mortality")
        pos_weight_tensor = pw * args.pos_weight_scale

    print(f"Building RETAIN model (mode={mode}),  variant={variant}) ...")
    model = RETAINModel(dataset=sample_dataset, **ABLATION_MODES[mode], **ADDITIONAL_FLAGS[variant],pos_weight=pos_weight_tensor,)

    print("Creating trainer...")
    trainer = Trainer(model=model, device=device)

    print("Training RETAIN mortality model...")
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
    pw_tag = f"_pw{args.pos_weight_scale}" if args.pos_weight else ""
    output_path = results_dir / f"retain_mortality_{mode}_{variant}{pw_tag}_test_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Saved test metrics to: {output_path}")


if __name__ == "__main__":
    main()
