# -*- coding: utf-8 -*-
"""Classification metrics without sklearn dependency."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict:
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_t, y_p):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / max(total, 1))
    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []
    for i in range(num_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        per_class[names[i]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    supports_arr = np.asarray(supports, dtype=np.float64)
    weights = supports_arr / max(float(supports_arr.sum()), 1.0)
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_precision": float(np.sum(np.asarray(precisions) * weights)),
        "weighted_recall": float(np.sum(np.asarray(recalls) * weights)),
        "weighted_f1": float(np.sum(np.asarray(f1s) * weights)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
