from pathlib import Path
from collections import Counter
import yaml

from pyhealth.datasets import MIMIC3Dataset, split_by_patient
from pyhealth.tasks.mortality_prediction import MortalityPredictionMIMIC3


def load_common_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "common.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_mimic3_base_dataset(config: dict) -> MIMIC3Dataset:
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / Path(config["dataset_root"])

    base_dataset = MIMIC3Dataset(
        root=str(dataset_root),
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    )
    return base_dataset


def build_mortality_dataset(config: dict):
    base_dataset = build_mimic3_base_dataset(config)
    mortality_task = MortalityPredictionMIMIC3()
    mortality_dataset = base_dataset.set_task(mortality_task)
    return mortality_dataset


def split_mortality_dataset(mortality_dataset, config: dict):
    split_cfg = config["split"]
    ratios = [split_cfg["train"], split_cfg["val"], split_cfg["test"]]
    seed = config["seed"]

    train_dataset, val_dataset, test_dataset = split_by_patient(
        mortality_dataset,
        ratios=ratios,
        seed=seed,
    )
    return train_dataset, val_dataset, test_dataset


def get_binary_label_counts(dataset, label_key: str = "mortality") -> dict:
    labels = []

    for i in range(len(dataset)):
        value = dataset[i][label_key]

        if hasattr(value, "item"):
            labels.append(int(value.item()))
        else:
            labels.append(int(value[0]))

    return dict(Counter(labels))


def get_mortality_data_splits(config_path: str = None):
    config = load_common_config(config_path)
    mortality_dataset = build_mortality_dataset(config)
    train_dataset, val_dataset, test_dataset = split_mortality_dataset(
        mortality_dataset, config
    )
    return mortality_dataset, train_dataset, val_dataset, test_dataset


def get_mortality_split_summary(config_path: str = None) -> dict:
    mortality_dataset, train_dataset, val_dataset, test_dataset = get_mortality_data_splits(
        config_path
    )

    summary = {
        "total_samples": len(mortality_dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_label_counts": get_binary_label_counts(train_dataset),
        "val_label_counts": get_binary_label_counts(val_dataset),
        "test_label_counts": get_binary_label_counts(test_dataset),
    }
    return summary