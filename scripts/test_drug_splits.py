import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.drug_recommendation import get_drug_recommendation_split_summary


def main():
    summary = get_drug_recommendation_split_summary()

    print("Drug recommendation split summary:")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Train samples: {summary['train_samples']}")
    print(f"Validation samples: {summary['val_samples']}")
    print(f"Test samples: {summary['test_samples']}")
    print()

    print("First train sample:")
    print(summary["first_train_sample"])


if __name__ == "__main__":
    main()