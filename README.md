# ITBA Neural Networks

These assignments use TF-IDF features to classify forum posts from the 20 Newsgroups dataset. They progress from a linear baseline to a multilayer perceptron (MLP) and then examine how regularization changes its behavior.

## Assignments

### TP1 · Softmax Regression

[Open the TP1 notebook](TP1/01_Softmax_Regression_20Newsgroups.ipynb)

Trains a linear classifier on TF-IDF vectors. Optuna compares text preprocessing and training settings. The follow-up compares three learning rates using validation curves and the measured time to a common accuracy target. The validation curve is then used to choose the training duration for a separate test evaluation.

### TP3 · Regularization

[Open the TP3 notebook](TP3/03_Regularization_20Newsgroups.ipynb)

Compares a baseline MLP with L1, L2, combined L1 and L2, dropout, and BatchNorm. The analysis looks at validation performance, learning curves, weight magnitudes, and training time. The follow-up tests two weaker L1 penalties to see whether the model can still classify unseen posts when more weights are close to zero.

## Run the Notebooks

Use Python 3.10 or later. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open a notebook, select the environment where you installed the requirements as its kernel, and run the cells in order. The setup downloads the 20 Newsgroups dataset and the NLTK resources on first use. The notebooks contain saved results for reference. Runtime and scores can vary with the machine and run settings.

The TP1 and TP3 notebooks contain all their preprocessing and training code.

