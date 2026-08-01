"""Serializable estimator wrappers for development-only gain ablations."""

from __future__ import annotations

from typing import Any

import numpy as np


class ProbabilityMeanClassifier:
    """Expose a classifier posterior mean as its continuous gain prediction."""

    def __init__(self, classifier: Any) -> None:
        self.classifier = classifier
        self.classes_ = classifier.classes_

    def predict_proba(self, features: Any) -> np.ndarray:
        return np.asarray(self.classifier.predict_proba(features), dtype=np.float64)

    def predict(self, features: Any) -> np.ndarray:
        probabilities = self.predict_proba(features)
        classes = np.asarray(self.classes_, dtype=np.float64)
        return probabilities @ classes


class LowerBoundGuardedProbabilityMeanClassifier:
    """Use a posterior mean only above a visible discrete-gain threshold."""

    def __init__(self, classifier: Any, discrete_threshold: float) -> None:
        self.classifier = classifier
        self.classes_ = classifier.classes_
        self.discrete_threshold = float(discrete_threshold)

    def predict_proba(self, features: Any) -> np.ndarray:
        return np.asarray(self.classifier.predict_proba(features), dtype=np.float64)

    def predict(self, features: Any) -> np.ndarray:
        probabilities = self.predict_proba(features)
        classes = np.asarray(self.classes_, dtype=np.float64)
        discrete = classes[np.argmax(probabilities, axis=1)]
        continuous = probabilities @ classes
        return np.where(discrete >= self.discrete_threshold, continuous, discrete)
