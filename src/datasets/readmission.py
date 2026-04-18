from pathlib import Path
from collections import Counter
from datetime import timedelta
import yaml

from pyhealth.datasets import MIMIC3Dataset, split_by_patient
from pyhealth.tasks.readmission_prediction import ReadmissionPredictionMIMIC3


def load_common_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "common.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_mimic3_base_dataset(config: dict) -> MIMIC3Dataset:
    dataset_root = Path(config["dataset_root"])

    base_dataset = MIMIC3Dataset(
        root=str(dataset_root),
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    )
    return base_dataset


def build_readmission_dataset(config: dict):
    base_dataset = build_mimic3_base_dataset(config)

    window_days = config["readmission"]["window_days"]
    readmission_task = ReadmissionPredictionMIMIC3(
        window=timedelta(days=window_days)
    )

    readmission_dataset = base_dataset.set_task(readmission_task)
    return readmission_dataset


def split_readmission_dataset(readmission_dataset, config: dict):
    split_cfg = config["split"]
    ratios = [split_cfg["train"], split_cfg["val"], split_cfg["test"]]
    seed = config["seed"]

    train_dataset, val_dataset, test_dataset = split_by_patient(
        readmission_dataset,
        ratios=ratios,
        seed=seed,
    )
    return train_dataset, val_dataset, test_dataset


def get_binary_label_counts(dataset, label_key: str = "readmission") -> dict:
    labels = []

    for i in range(len(dataset)):
        value = dataset[i][label_key]

        if hasattr(value, "item"):
            labels.append(int(value.item()))
        else:
            labels.append(int(value[0]))

    return dict(Counter(labels))


def get_readmission_data_splits(config_path: str = None):
    config = load_common_config(config_path)
    readmission_dataset = build_readmission_dataset(config)
    train_dataset, val_dataset, test_dataset = split_readmission_dataset(
        readmission_dataset, config
    )
    return readmission_dataset, train_dataset, val_dataset, test_dataset


def get_readmission_split_summary(config_path: str = None) -> dict:
    readmission_dataset, train_dataset, val_dataset, test_dataset = get_readmission_data_splits(
        config_path
    )

    summary = {
        "total_samples": len(readmission_dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_label_counts": get_binary_label_counts(train_dataset),
        "val_label_counts": get_binary_label_counts(val_dataset),
        "test_label_counts": get_binary_label_counts(test_dataset),
    }
    return summary