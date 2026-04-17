import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.readmission import get_readmission_split_summary


def main():
    summary = get_readmission_split_summary()

    print("Readmission split summary:")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Train samples: {summary['train_samples']}")
    print(f"Validation samples: {summary['val_samples']}")
    print(f"Test samples: {summary['test_samples']}")
    print()

    print(f"Train label counts: {summary['train_label_counts']}")
    print(f"Validation label counts: {summary['val_label_counts']}")
    print(f"Test label counts: {summary['test_label_counts']}")


if __name__ == "__main__":
    main()