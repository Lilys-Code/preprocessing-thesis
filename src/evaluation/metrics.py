import time
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test=None, class_names=None):
    """Evaluate a trained model and return classification metrics alongside a text report.

    Accepts either a Keras data generator (when `y_test` is None) or a plain
    array pair (`X_test`, `y_test`). Predictions are made in one pass and then
    compared against the true labels to compute accuracy, weighted precision,
    recall, F1, per-class F1, a confusion matrix, and total inference time.

    Returns a (metrics_dict, classification_report_str) tuple. The metrics dict
    is structured so it can be written directly to the CSV/JSON results log.
    """
    start_time = time.time()

    if y_test is None and hasattr(X_test, "classes"):
        y_true = X_test.classes
        y_pred_probs = model.predict(X_test, verbose=0)
        if class_names is None and hasattr(X_test, "class_indices"):
            class_names = sorted(X_test.class_indices, key=X_test.class_indices.get)
    else:
        y_true = y_test
        y_pred_probs = model.predict(X_test)

    inference_time = time.time() - start_time

    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "inference_time_sec": inference_time,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Per-class F1 scores are recorded for a more detailed breakdown in the results log
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    if class_names is not None:
        metrics["per_class_f1"] = dict(zip(class_names, per_class_f1.tolist()))
    else:
        metrics["per_class_f1"] = per_class_f1.tolist()

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return metrics, report
