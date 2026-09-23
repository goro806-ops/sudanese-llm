"""Evaluation Metrics for Multi-Regional Sudanese Models."""
from typing import List, Dict

def evaluate_dialect_accuracy(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate accuracy and dialect coverage score."""
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    total = max(len(targets), 1)
    return {
        "accuracy": correct / total,
        "sample_count": float(total)
    }
