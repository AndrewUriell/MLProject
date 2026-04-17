# MLProject

## Repository Structure
```text
project/
├── README.md
├── .gitignore
├── data/                    # actual MIMIC-III files, ignored by git
│   └── physionet.org/
├── configs/                 # experiment/config files
├── src/                     # all project code
│   ├── datasets/            # code to load MIMIC-III / PyHealth datasets
│   ├── preprocessing/       # code to clean data, build inputs, create splits
│   ├── models/              # baseline models + temporal transformer
│   ├── tasks/               # task-specific train/eval logic
│   └── utils/               # shared helper functions
├── scripts/                 # runnable entry-point scripts
├── logs/                    # training logs
├── checkpoints/             # saved model weights
├── results/                 # final metrics, tables, exported outputs
└── notes/                   # experiment notes / planning notes
```