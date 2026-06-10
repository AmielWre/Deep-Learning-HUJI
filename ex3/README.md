# Exercise 3: Sentiment Analysis

This folder contains a compact, object-oriented implementation of the IMDB
sentiment assignment. The code is split into five Python files:

- `config.py`: all paths, hyperparameters, device settings, and custom
  diagnostic reviews.
- `dataset.py`: CSV loading, text cleaning, tokenization, padding, masks, and
  vectorization. It uses GloVe through `torchtext` when available and otherwise
  falls back to deterministic offline token vectors.
- `models.py`: `ExRNN`, `ExGRU`, `ExMLP`, and `ExLRestSelfAtten`.
- `utils.py`: plotting and review-level diagnostics.
- `main.py`: isolated execution blocks for Task 1, Task 2, and Task 3/4.

To run an experiment, edit the final block in `main.py` and uncomment the task
call you want.

```powershell
cd ex3
..\.venv\Scripts\python.exe main.py
```

Plots and saved model weights are written to `ex3/outputs`. The diagnostic
reviews in `config.py` include TP, TN, FP, FN, and negation-shift examples such
as "not boring" and "not good" for comparing the isolated MLP against the
contextual attention model.
