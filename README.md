# ITBA Neural Networks

Experiments on text classification with the 20 Newsgroups dataset.

| Assignment | Notebook |
| --- | --- |
| TP1 · Softmax regression | [Open notebook](TP1/01_Softmax_Regression_20Newsgroups.ipynb) |

The reusable Python package lives under `src/tng/`. Its `data.py` module handles
text preprocessing, TF-IDF and data loading. Its `training.py` module provides
the softmax model and shared PyTorch training and evaluation routines.
Each notebook imports from `tng` and defines its own experiments, plots and analysis.

## Run TP1

Use Python 3.10 or later. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open the TP1 notebook, select the Python environment you just installed, and run
the cells in order. The first run downloads 20 Newsgroups and the required NLTK
resources. It runs 15 search trials and three timed validation experiments.
TensorBoard logs and the local MLflow database are generated under `runs/`.

