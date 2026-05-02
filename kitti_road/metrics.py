from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np
import torch

from .datasets import IGNORE_INDEX


def binary_counts(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, int]:
    pred = prob >= threshold
    valid = target != IGNORE_INDEX
    truth = target == 1
    pred = pred[valid]
    truth = truth[valid]
    tp = int((pred & truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())
    tn = int((~pred & ~truth).sum().item())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "precision": precision,
        "recall": recall,
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
        "f1": (2 * precision * recall) / (precision + recall + eps),
    }


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order].astype(np.float64)
    positives = labels.sum()
    if positives <= 0:
        return 0.0
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positives
    recall_steps = np.concatenate([[0.0], recall, [1.0]])
    precision_steps = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(precision_steps) - 2, -1, -1):
        precision_steps[i] = max(precision_steps[i], precision_steps[i + 1])
    changed = np.where(recall_steps[1:] != recall_steps[:-1])[0]
    return float(np.sum((recall_steps[changed + 1] - recall_steps[changed]) * precision_steps[changed + 1]))


def threshold_sweep(scores: np.ndarray, labels: np.ndarray, thresholds: Iterable[float] | None = None) -> list[Dict[str, float]]:
    thresholds = list(thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19))
    rows = []
    for threshold in thresholds:
        pred = scores >= threshold
        truth = labels == 1
        counts = {
            "tp": int((pred & truth).sum()),
            "fp": int((pred & ~truth).sum()),
            "fn": int((~pred & truth).sum()),
            "tn": int((~pred & ~truth).sum()),
        }
        row = {"threshold": round(float(threshold), 4), **metrics_from_counts(counts)}
        rows.append(row)
    return rows


@dataclass
class MetricAccumulator:
    threshold: float = 0.5
    counts: Dict[str, int] = field(default_factory=lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    scenario_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    scores: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)

    def update(self, logits: torch.Tensor, target: torch.Tensor, scenarios: list[str] | tuple[str, ...]) -> None:
        probs = torch.sigmoid(logits.squeeze(1)).detach().cpu()
        target_cpu = target.detach().cpu()
        for idx in range(probs.shape[0]):
            scenario = str(scenarios[idx])
            counts = binary_counts(probs[idx], target_cpu[idx], self.threshold)
            for key, value in counts.items():
                self.counts[key] += value
            bucket = self.scenario_counts.setdefault(scenario, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            for key, value in counts.items():
                bucket[key] += value
            valid = target_cpu[idx] != IGNORE_INDEX
            if bool(valid.any()):
                self.scores.append(probs[idx][valid].numpy().reshape(-1))
                self.labels.append((target_cpu[idx][valid].numpy().reshape(-1) == 1).astype(np.uint8))

    def compute(self) -> Dict[str, float]:
        metrics = metrics_from_counts(self.counts)
        if self.scores:
            scores = np.concatenate(self.scores)
            labels = np.concatenate(self.labels)
            metrics["ap"] = average_precision(scores, labels)
            sweep = threshold_sweep(scores, labels)
            metrics["max_f"] = max(row["f1"] for row in sweep)
        else:
            metrics["ap"] = 0.0
            metrics["max_f"] = 0.0
        return metrics

    def per_scenario(self) -> Dict[str, Dict[str, float]]:
        return {name: metrics_from_counts(counts) for name, counts in sorted(self.scenario_counts.items())}
