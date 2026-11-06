"""Time-based split, error metrics, peak detection and real-vs-predicted plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dengue.config import FIGURES_DIR, TARGET


def time_split(df, test_weeks=26):
    """Split by time, never at random: train = earlier weeks, test = latest weeks."""
    if test_weeks >= len(df):
        raise ValueError(f"test_weeks={test_weeks} leaves no training data ({len(df)} weeks)")
    return df.iloc[:-test_weeks], df.iloc[-test_weeks:]


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def peak_threshold(train, quantile=0.9):
    """An outbreak week is one above the 90th percentile of the training weeks."""
    return float(train[TARGET].quantile(quantile))


def peak_detection(y_true, y_pred, threshold):
    """Does the model see outbreaks coming? Recall/precision on weeks above the threshold."""
    real, pred = np.asarray(y_true) > threshold, np.asarray(y_pred) > threshold
    hits = int((real & pred).sum())
    return {
        "peak_recall": hits / real.sum() if real.any() else np.nan,
        "peak_precision": hits / pred.sum() if pred.any() else np.nan,
    }


def evaluate(models, train, test, quantile=0.9):
    """Fit every model on train, score on test. Returns (metrics table, predictions)."""
    threshold = peak_threshold(train, quantile)
    rows, predictions = [], pd.DataFrame({"real": test[TARGET]}, index=test.index)
    for model in models:
        pred = model.fit(train).predict(test)
        predictions[model.name] = pred
        rows.append({"model": model.name, "MAE": mae(test[TARGET], pred), "RMSE": rmse(test[TARGET], pred),
                     **peak_detection(test[TARGET], pred, threshold)})
    return pd.DataFrame(rows).set_index("model").sort_values("MAE"), predictions


def plot_real_vs_predicted(predictions, train=None, path=None):
    """Line plot of real cases vs each model's forecast on the test weeks."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if train is not None:
        ax.plot(train.index, train[TARGET], color="0.7", label="train")
    ax.plot(predictions.index, predictions["real"], color="black", linewidth=2, label="real")
    for col in predictions.columns.drop("real"):
        ax.plot(predictions.index, predictions[col], linewidth=1.3, label=col)
    ax.set_xlabel("Epidemiological week")
    ax.set_ylabel("Dengue cases")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    if path is not None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    return fig
