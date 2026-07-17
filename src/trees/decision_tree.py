"""Decision tree classifier built from scratch with numpy."""

from __future__ import annotations

import math

import numpy as np

from .node import Node
from .impurity import impurity
from .splitter import find_best_split
from .stopping import should_stop


class DecisionTree:
    """Decision tree classifier.

    Args:
        max_depth: Max tree depth (None for unlimited).
        min_samples_split: Min samples required to split.
        criterion: "gini" or "entropy".
        max_features: Feature sub-sampling ("sqrt", "log2", int, or None).
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: int | str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self._root: Node | None = None
        self._n_features: int | None = None
        self._classes: np.ndarray | None = None
        # Reuse single RNG instance for reproducibility across recursive calls
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "DecisionTree":
        """Train the tree. Returns self for chaining."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)

        self._n_features = X.shape[1]
        self._classes = np.unique(y)
        self._root = self._build_tree(X, y, sample_weight, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int) -> Node:
        """Build tree recursively."""
        node = Node(
            samples=len(y),
            value=self._weighted_class_counts(y, w),
            impurity_value=impurity(y, w, self.criterion),
            depth=depth,
        )

        if should_stop(depth, len(y), y, self.max_depth, self.min_samples_split):
            return node

        feature_indices = self._select_features()
        split = find_best_split(X, y, w, self.criterion, feature_indices)
        if split is None:
            return node

        left_mask = X[:, split.feature_index] <= split.threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return node  # degenerate split guard

        node.feature_index = split.feature_index
        node.threshold = split.threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], w[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], w[right_mask], depth + 1)
        return node

    def _weighted_class_counts(self, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        assert self._classes is not None
        return np.array([w[y == c].sum() for c in self._classes])

    def _select_features(self) -> np.ndarray:
        """Select features to search (supports max_features subsampling)."""
        n = self._n_features
        assert n is not None

        if self.max_features is None:
            return np.arange(n)
        if self.max_features == "sqrt":
            k = int(math.floor(math.sqrt(n)))
        elif self.max_features == "log2":
            k = int(math.floor(math.log2(n)))
        elif isinstance(self.max_features, int):
            k = self.max_features
        else:
            raise ValueError(
                f"max_features must be int, 'sqrt', 'log2', or None; got {self.max_features!r}"
            )

        k = max(1, min(k, n))
        return self._rng.choice(n, size=k, replace=False)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each sample."""
        if self._root is None:
            raise RuntimeError("Call fit() before predict_proba()")
        X = np.asarray(X, dtype=float)
        probs = np.array([self._traverse(x, self._root).value for x in X])
        row_sums = probs.sum(axis=1, keepdims=True)
        result: np.ndarray = probs / row_sums
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        assert self._classes is not None
        proba = self.predict_proba(X)
        result: np.ndarray = self._classes[np.argmax(proba, axis=1)]
        return result

    def _traverse(self, x: np.ndarray, node: Node) -> Node:
        """Walk from the root to the leaf reached by a single sample x."""
        while not node.is_leaf:
            assert node.feature_index is not None and node.threshold is not None
            assert node.left is not None and node.right is not None
            node = node.left if x[node.feature_index] <= node.threshold else node.right
        return node

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def depth(self) -> int:
        """Max depth of the tree."""

        def _max_depth(node: Node) -> int:
            if node.is_leaf:
                return node.depth
            assert node.left is not None and node.right is not None
            return max(_max_depth(node.left), _max_depth(node.right))

        return _max_depth(self._root) if self._root else 0

    @property
    def n_leaves(self) -> int:
        """Count of leaf nodes."""

        def _count(node: Node) -> int:
            if node.is_leaf:
                return 1
            assert node.left is not None and node.right is not None
            return _count(node.left) + _count(node.right)

        return _count(self._root) if self._root else 0

    def feature_importances(self) -> np.ndarray:
        """Normalized mean decrease in impurity per feature."""
        if self._root is None:
            raise RuntimeError("Call fit() before feature_importances()")
        assert self._n_features is not None

        raw = np.zeros(self._n_features)
        self._accumulate_importance(self._root, raw)
        total = raw.sum()
        if total <= 0:
            return raw
        result: np.ndarray = raw / total
        return result

    def _accumulate_importance(self, node: Node, raw: np.ndarray) -> None:
        """Accumulate feature importance: n * (parent - weighted_child_impurity)."""
        if node.is_leaf:
            return

        assert node.left is not None and node.right is not None
        n = node.samples
        weighted_child_impurity = (
            (node.left.samples / n) * node.left.impurity_value
            + (node.right.samples / n) * node.right.impurity_value
        )
        delta_impurity = node.impurity_value - weighted_child_impurity
        raw[node.feature_index] += n * delta_impurity

        self._accumulate_importance(node.left, raw)
        self._accumulate_importance(node.right, raw)

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        """Text representation of tree (suppressed if depth > 4)."""
        if self._root is None:
            return "DecisionTree(unfitted)"
        if self.depth > 4:
            return (
                f"DecisionTree(depth={self.depth}, n_leaves={self.n_leaves}, "
                f"[repr suppressed: depth > 4])"
            )
        lines: list[str] = []
        self._repr_node(self._root, prefix="", lines=lines)
        return "\n".join(lines)

    def _repr_node(self, node: Node, prefix: str, lines: list[str]) -> None:
        assert self._classes is not None
        dist = ", ".join(f"{c}:{v:.1f}" for c, v in zip(self._classes, node.value))
        if node.is_leaf:
            lines.append(
                f"{prefix}Leaf(samples={node.samples}, {self.criterion}={node.impurity_value:.3f}, "
                f"dist=[{dist}])"
            )
        else:
            lines.append(
                f"{prefix}[X{node.feature_index} <= {node.threshold:.3f}] "
                f"(samples={node.samples}, {self.criterion}={node.impurity_value:.3f}, dist=[{dist}])"
            )
            assert node.left is not None and node.right is not None
            self._repr_node(node.left, prefix + "  L-- ", lines)
            self._repr_node(node.right, prefix + "  R-- ", lines)


class DecisionStump(DecisionTree):
    """Decision tree with max_depth=1 (weak learner for boosting)."""

    def __init__(self, criterion: str = "gini", random_state: int | None = None) -> None:
        super().__init__(max_depth=1, criterion=criterion, random_state=random_state)


