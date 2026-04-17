from pathlib import Path
import yaml

from pyhealth.datasets import MIMIC3Dataset, split_by_patient
from pyhealth.tasks.drug_recommendation import DrugRecommendationMIMIC3


def load_common_config(config_path: str = "configs/common.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_mimic3_base_dataset(config: dict) -> MIMIC3Dataset:
    dataset_root = Path(config["dataset_root"])

    base_dataset = MIMIC3Dataset(
        root=str(dataset_root),
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    )
    return base_dataset


def build_drug_recommendation_dataset(config: dict):
    base_dataset = build_mimic3_base_dataset(config)

    # Keeping the default task config for now so we can verify the baseline pipeline.
    # We can revisit code mapping / vocabulary filtering after the baseline is running.
    drug_task = DrugRecommendationMIMIC3()
    drug_dataset = base_dataset.set_task(drug_task)
    return drug_dataset


def split_drug_recommendation_dataset(drug_dataset, config: dict):
    split_cfg = config["split"]
    ratios = [split_cfg["train"], split_cfg["val"], split_cfg["test"]]
    seed = config["seed"]

    train_dataset, val_dataset, test_dataset = split_by_patient(
        drug_dataset,
        ratios=ratios,
        seed=seed,
    )
    return train_dataset, val_dataset, test_dataset


def get_drug_recommendation_data_splits(config_path: str = "configs/common.yaml"):
    config = load_common_config(config_path)
    drug_dataset = build_drug_recommendation_dataset(config)
    train_dataset, val_dataset, test_dataset = split_drug_recommendation_dataset(
        drug_dataset, config
    )
    return drug_dataset, train_dataset, val_dataset, test_dataset


def get_drug_recommendation_split_summary(config_path: str = "configs/common.yaml") -> dict:
    drug_dataset, train_dataset, val_dataset, test_dataset = get_drug_recommendation_data_splits(
        config_path
    )

    summary = {
        "total_samples": len(drug_dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "first_train_sample": train_dataset[0] if len(train_dataset) > 0 else None,
    }
    return summary