from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def macro_specificity(y_true, y_pred, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    total = cm.sum()
    values = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        values.append(tn / max(tn + fp, 1))
    return float(np.mean(values))


def classification_metrics(y_true, y_pred, num_classes: int):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "specificity_macro": macro_specificity(y_true, y_pred, num_classes),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
