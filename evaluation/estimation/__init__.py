"""Relative-pose evaluation across the six paper test sets."""

from .metrics import DATASETS, error_auc, pose_error_vector

__all__ = ["DATASETS", "error_auc", "pose_error_vector"]
