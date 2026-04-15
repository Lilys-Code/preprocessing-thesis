import time
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test=None):
    start_time = time.time()

    if y_test is None and hasattr(X_test, 'classes'):
        y_true = X_test.classes
        y_pred_probs = model.predict(X_test, verbose=0)
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
        "inference_time_sec": inference_time
    }

    report = classification_report(y_true, y_pred)

    return metrics, report