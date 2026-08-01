import numpy as np

from active_diagnosis_v13.gain_estimators import (
    LowerBoundGuardedProbabilityMeanClassifier,
    ProbabilityMeanClassifier,
)


class _Classifier:
    classes_ = np.asarray([0.5, 1.0, 1.5])

    def predict_proba(self, features):
        return np.asarray([[0.2, 0.3, 0.5] for _ in features])


def test_probability_mean_classifier_returns_continuous_gain() -> None:
    model = ProbabilityMeanClassifier(_Classifier())
    assert np.allclose(model.predict([[1.0], [2.0]]), [1.15, 1.15])
    assert model.predict_proba([[1.0]]).shape == (1, 3)


class _GuardClassifier:
    classes_ = np.asarray([0.5, 1.0, 1.5])

    def predict_proba(self, features):
        return np.asarray(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.5, 0.3],
            ][: len(features)]
        )


def test_lower_bound_guard_keeps_low_discrete_and_smooths_high_discrete() -> None:
    model = LowerBoundGuardedProbabilityMeanClassifier(
        _GuardClassifier(), discrete_threshold=1.0
    )
    assert np.allclose(model.predict([[1.0], [2.0]]), [0.5, 1.05])
    assert model.predict_proba([[1.0]]).shape == (1, 3)
