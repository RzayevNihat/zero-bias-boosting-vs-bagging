"""AdaBoost variants with weighted decision stumps, from scratch.

This module implements the multi-class discrete SAMME algorithm of Zhu et al.,
which reduces to the classic Freund and Schapire AdaBoost rule when there are
two classes.  It also includes a SAMME.R bonus variant that uses real-valued
class-probability scores from smoothed probability stumps.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np

from src.boosting._validation import validate_inputs as _validate_inputs

EPS = 1e-10


class DecisionStump:
    """Depth-1 decision tree supporting weighted samples.

    The stump searches every midpoint between consecutive distinct values
    of every feature. For each candidate split it computes weighted class
    counts on the left and right child and chooses the split with minimum
    weighted impurity.
    """

    def __init__(
        self,
        criterion: str = "gini",
        random_state: Optional[int] = None,
    ) -> None:
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be either 'gini' or 'entropy'.")
        self.criterion = criterion
        self.random_state = random_state
        self.feature_index_: Optional[int] = None
        self.threshold_: Optional[float] = None
        self.left_class_: Optional[int] = None
        self.right_class_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "DecisionStump":
        """Fit the best weighted single split on ``X`` and ``y``."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        _validate_inputs(X, y)
        n_samples, n_features = X.shape

        if sample_weight is None:
            sample_weight = np.full(n_samples, 1.0 / n_samples)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != (n_samples,):
                raise ValueError("sample_weight must have shape (n_samples,).")
            if np.any(sample_weight < 0.0):
                raise ValueError("sample_weight must be non-negative.")
            weight_sum = float(sample_weight.sum())
            if weight_sum <= EPS:
                raise ValueError("sample_weight must contain positive total weight.")
            sample_weight = sample_weight / weight_sum

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        n_classes = self.classes_.size

        class_weights = np.bincount(
            y_encoded, weights=sample_weight, minlength=n_classes
        )
        majority = int(np.argmax(class_weights))
        self.feature_index_ = None
        self.threshold_ = None
        self.left_class_ = majority
        self.right_class_ = majority

        best_impurity = np.inf
        rng = np.random.default_rng(self.random_state)

        for feature in rng.permutation(n_features):
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]

            one_hot = np.zeros((n_samples, n_classes), dtype=float)
            one_hot[np.arange(n_samples), y_encoded[order]] = sample_weight[order]
            left_counts_all = np.cumsum(one_hot, axis=0)
            total_counts = left_counts_all[-1]

            valid = values[:-1] < values[1:]
            if not np.any(valid):
                continue

            left_counts = left_counts_all[:-1][valid]
            right_counts = total_counts - left_counts
            left_totals = left_counts.sum(axis=1)
            right_totals = right_counts.sum(axis=1)

            if self.criterion == "gini":
                safe_left = np.maximum(left_totals, EPS)
                safe_right = np.maximum(right_totals, EPS)
                left_impurity = 1.0 - np.sum(
                    (left_counts / safe_left[:, None]) ** 2, axis=1
                )
                right_impurity = 1.0 - np.sum(
                    (right_counts / safe_right[:, None]) ** 2, axis=1
                )
            else:
                left_prob = left_counts / np.maximum(left_totals[:, None], EPS)
                right_prob = right_counts / np.maximum(right_totals[:, None], EPS)
                left_impurity = -np.sum(
                    np.where(left_prob > 0.0, left_prob * np.log2(left_prob), 0.0),
                    axis=1,
                )
                right_impurity = -np.sum(
                    np.where(right_prob > 0.0, right_prob * np.log2(right_prob), 0.0),
                    axis=1,
                )

            impurity = left_totals * left_impurity + right_totals * right_impurity
            best_local = int(np.argmin(impurity))
            if impurity[best_local] < best_impurity:
                best_impurity = float(impurity[best_local])
                split_position = np.flatnonzero(valid)[best_local]
                self.feature_index_ = int(feature)
                self.threshold_ = float(
                    (values[split_position] + values[split_position + 1]) / 2.0
                )
                self.left_class_ = int(np.argmax(left_counts[best_local]))
                self.right_class_ = int(np.argmax(right_counts[best_local]))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by routing samples through the stump."""
        if self.classes_ is None:
            raise RuntimeError("DecisionStump must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        _validate_inputs(X)

        if self.feature_index_ is None:
            return np.full(X.shape[0], self.classes_[self.left_class_])

        goes_left = X[:, self.feature_index_] <= self.threshold_
        encoded = np.where(goes_left, self.left_class_, self.right_class_)
        return self.classes_[encoded]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return hard class probabilities induced by the stump prediction."""
        predictions = self.predict(X)
        proba = np.zeros((predictions.shape[0], self.classes_.size), dtype=float)
        encoded = np.searchsorted(self.classes_, predictions)
        proba[np.arange(predictions.shape[0]), encoded] = 1.0
        return proba


class AdaBoostClassifier:
    """Multi-class AdaBoost classifier using discrete SAMME voting.

    Attributes:
        estimators_: Fitted decision stumps.
        estimator_weights_: Alpha coefficient for each stump.
        estimator_errors_: Weighted training error for each stump.
        sample_weights_history_: Weight vector before round 1, after each
            completed round, and therefore length ``len(estimators_) + 1``.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        criterion: str = "gini",
        probability_method: str = "softmax",
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive.")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be either 'gini' or 'entropy'.")
        if probability_method not in {"softmax", "vote"}:
            raise ValueError("probability_method must be either 'softmax' or 'vote'.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.probability_method = probability_method
        self.random_state = random_state

        self.estimators_: List[DecisionStump] = []
        self.estimator_weights_: np.ndarray = np.array([], dtype=float)
        self.estimator_errors_: np.ndarray = np.array([], dtype=float)
        self.sample_weights_history_: List[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        """Fit the SAMME boosting loop and store per-round diagnostics."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        _validate_inputs(X, y)

        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.size)
        if self.n_classes_ < 2:
            raise ValueError("AdaBoost requires at least two classes in y.")

        n_samples = X.shape[0]
        y_encoded = np.searchsorted(self.classes_, y)
        weights = np.full(n_samples, 1.0 / n_samples, dtype=float)

        self.estimators_ = []
        alphas: List[float] = []
        errors: List[float] = []
        self.sample_weights_history_ = [weights.copy()]

        for round_index in range(self.n_estimators):
            seed = (
                None
                if self.random_state is None
                else int(self.random_state) + round_index
            )
            stump = DecisionStump(
                criterion=self.criterion,
                random_state=seed,
            ).fit(X, y_encoded, weights)

            predictions = stump.predict(X)
            incorrect = predictions != y_encoded
            error = float(np.dot(weights, incorrect))

            chance_error = 1.0 - 1.0 / self.n_classes_
            if error >= chance_error:
                if round_index == 0:
                    raise ValueError(
                        "First weak learner performs no better than chance; "
                        "boosting cannot proceed on this data."
                    )
                break

            clipped_error = float(np.clip(error, EPS, 1.0 - EPS))
            alpha = self.learning_rate * (
                np.log((1.0 - clipped_error) / clipped_error)
                + np.log(self.n_classes_ - 1.0)
            )

            weights = weights * np.exp(alpha * incorrect)
            weights = weights / max(float(weights.sum()), EPS)

            self.estimators_.append(stump)
            alphas.append(float(alpha))
            errors.append(clipped_error)
            self.sample_weights_history_.append(weights.copy())

        self.estimator_weights_ = np.array(alphas, dtype=float)
        self.estimator_errors_ = np.array(errors, dtype=float)
        return self

    def _vote_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return alpha-weighted class votes for each sample."""
        if not self.estimators_:
            raise RuntimeError("AdaBoostClassifier must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        _validate_inputs(X)

        votes = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        rows = np.arange(X.shape[0])
        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            votes[rows, stump.predict(X)] += alpha
        return votes

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels through the alpha-weighted majority vote."""
        votes = self._vote_matrix(X)
        return self.classes_[np.argmax(votes, axis=1)]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return alpha-weighted SAMME class scores."""
        return self._vote_matrix(X)

    def predict_proba(self, X: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """Return probability-like scores from the SAMME vote matrix.

        ``method="softmax"`` applies a stable softmax to the weighted class
        scores and is the default calibration-friendly choice. ``method="vote"``
        keeps the older normalized-vote interpretation for reproducibility.
        """
        method = self.probability_method if method is None else method
        if method not in {"softmax", "vote"}:
            raise ValueError("method must be either 'softmax' or 'vote'.")

        votes = self._vote_matrix(X)
        if method == "softmax":
            shifted = votes - votes.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), EPS)

        totals = np.maximum(votes.sum(axis=1, keepdims=True), EPS)
        return votes / totals

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predictions after every completed boosting round."""
        if not self.estimators_:
            raise RuntimeError("AdaBoostClassifier must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        _validate_inputs(X)

        votes = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        rows = np.arange(X.shape[0])
        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            votes[rows, stump.predict(X)] += alpha
            yield self.classes_[np.argmax(votes, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean classification accuracy."""
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))

    @property
    def estimator_weights(self) -> np.ndarray:
        """Alpha coefficients of fitted weak learners."""
        return self.estimator_weights_

    @property
    def estimator_errors(self) -> np.ndarray:
        """Weighted training errors of fitted weak learners."""
        return self.estimator_errors_


class ProbaDecisionStump(DecisionStump):
    """Decision stump whose leaves expose smoothed class probabilities.

    Discrete SAMME only needs each weak learner's hard class prediction.  SAMME.R
    needs per-class probabilities because its update uses log-probability
    contrasts.  This subclass keeps the same weighted split search as
    ``DecisionStump`` and stores a smoothed weighted class distribution for each
    leaf, avoiding exact zeros in ``log(p_k(x))``.
    """

    def __init__(
        self,
        criterion: str = "gini",
        random_state: Optional[int] = None,
        smoothing: float = 1e-3,
    ) -> None:
        if smoothing <= 0.0:
            raise ValueError("smoothing must be strictly positive.")
        super().__init__(criterion=criterion, random_state=random_state)
        self.smoothing = smoothing
        self.left_proba_: Optional[np.ndarray] = None
        self.right_proba_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ProbaDecisionStump":
        """Fit the stump and store smoothed leaf distributions."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        _validate_inputs(X, y)
        n_samples = X.shape[0]

        if sample_weight is None:
            sample_weight = np.full(n_samples, 1.0 / n_samples)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != (n_samples,):
                raise ValueError("sample_weight must have shape (n_samples,).")
            if np.any(sample_weight < 0.0):
                raise ValueError("sample_weight must be non-negative.")
            weight_sum = float(sample_weight.sum())
            if weight_sum <= EPS:
                raise ValueError("sample_weight must contain positive total weight.")
            sample_weight = sample_weight / weight_sum

        super().fit(X, y, sample_weight)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        n_classes = self.classes_.size

        if self.feature_index_ is None:
            counts = np.bincount(y_encoded, weights=sample_weight, minlength=n_classes)
            proba = (counts + self.smoothing) / (
                counts.sum() + n_classes * self.smoothing
            )
            self.left_proba_ = proba
            self.right_proba_ = proba
            return self

        goes_left = X[:, self.feature_index_] <= self.threshold_
        for mask, attr in ((goes_left, "left_proba_"), (~goes_left, "right_proba_")):
            counts = np.bincount(
                y_encoded[mask],
                weights=sample_weight[mask],
                minlength=n_classes,
            )
            proba = (counts + self.smoothing) / (
                counts.sum() + n_classes * self.smoothing
            )
            setattr(self, attr, proba)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return each sample's smoothed leaf class-probability distribution."""
        if self.classes_ is None or self.left_proba_ is None:
            raise RuntimeError("ProbaDecisionStump must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        _validate_inputs(X)

        if self.feature_index_ is None:
            return np.tile(self.left_proba_, (X.shape[0], 1))

        goes_left = X[:, self.feature_index_] <= self.threshold_
        proba = np.where(
            goes_left[:, None],
            self.left_proba_[None, :],
            self.right_proba_[None, :],
        )
        return proba


class AdaBoostSAMMERClassifier:
    """Multi-class AdaBoost using the real-valued SAMME.R update rule.

    SAMME.R uses weak-learner probability estimates directly.  At each round,
    leaf probabilities ``p_k(x)`` are transformed into centered log-probability
    contrasts and accumulated as a real-valued class score.  This is the bonus
    variant used for the multi-class calibration comparison.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        criterion: str = "gini",
        smoothing: float = 1e-3,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be either 'gini' or 'entropy'.")
        if smoothing <= 0.0:
            raise ValueError("smoothing must be strictly positive.")

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.smoothing = smoothing
        self.random_state = random_state

        self.estimators_: List[ProbaDecisionStump] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0

    def _h_scores(self, proba: np.ndarray) -> np.ndarray:
        """Convert class probabilities into centered SAMME.R scores."""
        log_p = np.log(np.clip(proba, EPS, 1.0))
        return (self.n_classes_ - 1.0) * (
            log_p - log_p.mean(axis=1, keepdims=True)
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostSAMMERClassifier":
        """Fit the SAMME.R boosting loop."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        _validate_inputs(X, y)

        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.size)
        if self.n_classes_ < 2:
            raise ValueError("AdaBoost requires at least two classes in y.")

        n_samples = X.shape[0]
        y_encoded = np.searchsorted(self.classes_, y)
        y_code = np.full(
            (n_samples, self.n_classes_),
            -1.0 / (self.n_classes_ - 1.0),
            dtype=float,
        )
        y_code[np.arange(n_samples), y_encoded] = 1.0

        weights = np.full(n_samples, 1.0 / n_samples, dtype=float)
        self.estimators_ = []

        for round_index in range(self.n_estimators):
            seed = (
                None
                if self.random_state is None
                else int(self.random_state) + round_index
            )
            stump = ProbaDecisionStump(
                criterion=self.criterion,
                random_state=seed,
                smoothing=self.smoothing,
            ).fit(X, y_encoded, weights)

            proba = stump.predict_proba(X)
            proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), EPS)
            h = self._h_scores(proba)

            weighted_margin = np.sum(y_code * h, axis=1)
            weights = weights * np.exp(
                -((self.n_classes_ - 1.0) / self.n_classes_) * weighted_margin
            )
            weights = weights / max(float(weights.sum()), EPS)

            self.estimators_.append(stump)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return accumulated real-valued class scores."""
        if not self.estimators_:
            raise RuntimeError("AdaBoostSAMMERClassifier must be fitted first.")
        X = np.asarray(X, dtype=float)
        _validate_inputs(X)

        scores = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for stump in self.estimators_:
            proba = stump.predict_proba(X)
            proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), EPS)
            scores += self._h_scores(proba)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels as the argmax of accumulated SAMME.R scores."""
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert accumulated SAMME.R scores to calibrated probabilities."""
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores / max(self.n_classes_ - 1.0, 1.0))
        return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), EPS)

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predictions after every completed SAMME.R round."""
        if not self.estimators_:
            raise RuntimeError("AdaBoostSAMMERClassifier must be fitted first.")
        X = np.asarray(X, dtype=float)
        _validate_inputs(X)

        scores = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for stump in self.estimators_:
            proba = stump.predict_proba(X)
            proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), EPS)
            scores += self._h_scores(proba)
            yield self.classes_[np.argmax(scores, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean classification accuracy."""
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))
