"""Binary Gradient Boosting with log-loss, implemented from scratch.

The model follows the classic additive boosting recipe for binary
classification.  It starts from the training-set log-odds, fits small CART
regression trees to the negative log-loss gradient, and adds each tree's
prediction to the current score function.

Only binary targets are supported because this module is used for the
Gradient Boosting bonus comparison against AdaBoost.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np

from src.boosting._validation import validate_inputs as _validate_inputs

EPS = 1e-10


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    clipped = np.clip(scores, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class _TreeNode:
    """Node used by the internal CART regression tree."""

    __slots__ = ("feature_index", "threshold", "left", "right", "value")

    def __init__(self, value: float) -> None:
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["_TreeNode"] = None
        self.right: Optional["_TreeNode"] = None
        self.value = value


class _RegressionTree:
    """Small CART regressor used as the GBM weak learner.

    Each split is chosen by exhaustive midpoint search, minimizing the sum of
    squared residuals.  The implementation is intentionally compact and mirrors
    the exact-search style used by the AdaBoost decision stump.
    """

    def __init__(
        self,
        max_depth: Optional[int] = 3,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
    ) -> None:
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be a positive integer or None.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root_: Optional[_TreeNode] = None

    def fit(self, X: np.ndarray, residual: np.ndarray) -> "_RegressionTree":
        """Fit a regression tree to the residual targets."""
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, residual, depth=0, rng=rng)
        return self

    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        rng: np.random.Generator,
    ) -> _TreeNode:
        node = _TreeNode(value=float(np.mean(y)) if y.size else 0.0)

        n_samples = X.shape[0]
        if n_samples < self.min_samples_split or n_samples < 2:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if np.allclose(y, y[0]):
            return node

        split = self._best_split(X, y, rng)
        if split is None:
            return node

        feature_index, threshold, left_mask = split
        if not np.any(left_mask) or np.all(left_mask):
            return node

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1, rng)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1, rng)
        return node

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> Optional[tuple[int, float, np.ndarray]]:
        n_samples, n_features = X.shape
        total_sum = float(y.sum())
        total_sq = float(np.sum(y**2))
        total_sse = total_sq - (total_sum**2) / n_samples

        best_gain = EPS
        best: Optional[tuple[int, float, np.ndarray]] = None

        for feature in rng.permutation(n_features):
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]
            y_sorted = y[order]

            valid = values[:-1] < values[1:]
            if not np.any(valid):
                continue

            cum_sum = np.cumsum(y_sorted)
            cum_sq = np.cumsum(y_sorted**2)
            left_n = np.arange(1, n_samples, dtype=float)
            left_sum = cum_sum[:-1]
            left_sq = cum_sq[:-1]
            right_n = n_samples - left_n
            right_sum = total_sum - left_sum
            right_sq = total_sq - left_sq

            left_sse = left_sq - (left_sum**2) / left_n
            right_sse = right_sq - (right_sum**2) / right_n
            sse = np.where(valid, left_sse + right_sse, np.inf)

            local_best = int(np.argmin(sse))
            if not np.isfinite(sse[local_best]):
                continue

            gain = total_sse - float(sse[local_best])
            if gain > best_gain:
                best_gain = gain
                threshold = float(
                    (values[local_best] + values[local_best + 1]) / 2.0
                )
                left_mask = X[:, feature] <= threshold
                best = (int(feature), threshold, left_mask)

        return best

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict a continuous value for each row."""
        if self.root_ is None:
            raise RuntimeError("_RegressionTree must be fitted before prediction.")
        return np.array([self._predict_one(row, self.root_) for row in X], dtype=float)

    def _predict_one(self, row: np.ndarray, node: _TreeNode) -> float:
        while node.feature_index is not None:
            assert node.threshold is not None
            if row[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node.value


class GradientBoostingClassifier:
    """Binary Gradient Boosting classifier trained with logistic log-loss.

    Attributes:
        estimators_: Regression trees fitted to residuals.
        init_prediction_: Initial constant log-odds score.
        classes_: The two labels seen during ``fit``; ``classes_[1]`` is the
            positive class for probabilities.
        train_log_loss_: Training log-loss after each boosting stage.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive.")
        if max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.estimators_: List[_RegressionTree] = []
        self.init_prediction_: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None
        self.train_log_loss_: np.ndarray = np.array([], dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        """Fit the additive log-loss boosting model."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        _validate_inputs(X, y)

        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(
                "GradientBoostingClassifier supports binary classification only, "
                f"got {self.classes_.size} classes."
            )

        y_encoded = (y == self.classes_[1]).astype(float)
        n_samples = X.shape[0]
        base_rate = float(np.clip(y_encoded.mean(), EPS, 1.0 - EPS))
        self.init_prediction_ = float(np.log(base_rate / (1.0 - base_rate)))

        scores = np.full(n_samples, self.init_prediction_, dtype=float)
        self.estimators_ = []
        losses: List[float] = []

        for stage in range(self.n_estimators):
            probabilities = _sigmoid(scores)
            residual = y_encoded - probabilities

            seed = None if self.random_state is None else int(self.random_state) + stage
            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=seed,
            ).fit(X, residual)

            scores = scores + self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            losses.append(self._binary_log_loss(y_encoded, _sigmoid(scores)))

        self.train_log_loss_ = np.array(losses, dtype=float)
        return self

    @staticmethod
    def _binary_log_loss(y_true: np.ndarray, probability: np.ndarray) -> float:
        """Return binary negative log-likelihood for internal diagnostics."""
        probability = np.clip(probability, EPS, 1.0 - EPS)
        loss = y_true * np.log(probability) + (1.0 - y_true) * np.log(
            1.0 - probability
        )
        return -float(np.mean(loss))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the raw additive score for the positive class."""
        if not self.estimators_ or self.init_prediction_ is None:
            raise RuntimeError(
                "GradientBoostingClassifier must be fitted before prediction."
            )
        X = np.asarray(X, dtype=float)
        _validate_inputs(X)

        scores = np.full(X.shape[0], self.init_prediction_, dtype=float)
        for tree in self.estimators_:
            scores = scores + self.learning_rate * tree.predict(X)
        return scores

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for the raw additive score."""
        return self.decision_function(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities in ``classes_`` order."""
        positive_probability = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - positive_probability, positive_probability])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels using the larger class probability."""
        if self.classes_ is None:
            raise RuntimeError(
                "GradientBoostingClassifier must be fitted before prediction."
            )
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def staged_predict_proba(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield class probabilities after each boosting stage."""
        if not self.estimators_ or self.init_prediction_ is None:
            raise RuntimeError(
                "GradientBoostingClassifier must be fitted before prediction."
            )
        X = np.asarray(X, dtype=float)
        _validate_inputs(X)

        scores = np.full(X.shape[0], self.init_prediction_, dtype=float)
        for tree in self.estimators_:
            scores = scores + self.learning_rate * tree.predict(X)
            positive_probability = _sigmoid(scores)
            yield np.column_stack([1.0 - positive_probability, positive_probability])

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predicted labels after each completed boosting stage."""
        if self.classes_ is None:
            raise RuntimeError(
                "GradientBoostingClassifier must be fitted before prediction."
            )
        for probabilities in self.staged_predict_proba(X):
            yield self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean classification accuracy."""
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))
