"""
Experiment 7: Unsupervised-learning analysis.

The experiment applies shared preprocessing, PCA, K-Means, and DBSCAN
to the handwritten-digits dataset and exports reproducible results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score

from src.unsupervised.dbscan import DBSCAN
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA
from src.utils.preprocessing import PreprocessingPipeline


RANDOM_STATE = 42
SAMPLE_LIMIT = 600
N_CLUSTERS = 10
PCA_COMPONENTS = 2

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def ensure_output_directories() -> None:
    """Create output directories when they do not already exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(
    sample_limit: int = SAMPLE_LIMIT,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a reproducible subset of the digits dataset.
    """
    dataset = load_digits()

    X = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target, dtype=int)

    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive.")

    sample_limit = min(sample_limit, X.shape[0])

    rng = np.random.default_rng(random_state)
    selected_indices = rng.choice(
        X.shape[0],
        size=sample_limit,
        replace=False,
    )

    return X[selected_indices], y[selected_indices]


def plot_pca_projection(
    X_reduced: np.ndarray,
    labels: np.ndarray,
) -> Path:
    """Save the two-dimensional PCA projection."""
    output_path = FIGURES_DIR / "unsupervised_pca_projection.png"

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=labels,
        s=18,
        alpha=0.75,
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Digits Dataset — PCA Projection")
    plt.colorbar(scatter, label="True digit")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def plot_scree_curve(
    pca: PCA,
) -> Path:
    """Save explained and cumulative variance curves."""
    output_path = FIGURES_DIR / "unsupervised_pca_scree.png"

    components, explained_variance, cumulative_variance = (
        pca.scree_data()
    )

    explained_ratio = pca.explained_variance_ratio_

    if explained_ratio is None:
        raise RuntimeError(
            "PCA did not produce explained variance ratios."
        )

    plt.figure(figsize=(8, 5))
    plt.plot(
        components,
        explained_ratio,
        marker="o",
        label="Explained variance ratio",
    )
    plt.plot(
        components,
        cumulative_variance,
        marker="s",
        label="Cumulative explained variance",
    )

    plt.xlabel("Principal component")
    plt.ylabel("Variance ratio")
    plt.title("PCA Scree Plot")
    plt.xticks(components)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def plot_elbow_curve(
    X: np.ndarray,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Calculate and save the K-Means elbow curve."""
    cluster_values, inertias = KMeans.elbow_curve(
        X,
        cluster_range=range(1, 13),
        n_init=5,
        random_state=RANDOM_STATE,
    )

    output_path = FIGURES_DIR / "unsupervised_kmeans_elbow.png"

    plt.figure(figsize=(8, 5))
    plt.plot(
        cluster_values,
        inertias,
        marker="o",
    )

    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow Curve")
    plt.xticks(cluster_values)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path, cluster_values, inertias


def plot_k_distance_curve(
    X: np.ndarray,
    k: int,
) -> tuple[Path, np.ndarray]:
    """Calculate and save the DBSCAN k-distance curve."""
    sample_order, distances = DBSCAN.k_distance_curve(
        X,
        k=k,
    )

    output_path = FIGURES_DIR / "unsupervised_dbscan_k_distance.png"

    plt.figure(figsize=(8, 5))
    plt.plot(
        sample_order,
        distances,
    )

    plt.xlabel("Samples ordered by distance")
    plt.ylabel(f"Distance to neighbor {k}")
    plt.title("DBSCAN k-Distance Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path, distances


def run_experiment(
    sample_limit: int = SAMPLE_LIMIT,
) -> dict[str, Any]:
    """
    Run the complete unsupervised-learning experiment.
    """
    ensure_output_directories()

    X, y = load_dataset(
        sample_limit=sample_limit,
        random_state=RANDOM_STATE,
    )

    pipeline = PreprocessingPipeline()
    X_scaled = pipeline.fit_transform(X)

    full_pca = PCA(
        n_components=min(
            10,
            X_scaled.shape[1],
        )
    )
    full_pca.fit(X_scaled)

    projection_pca = PCA(
        n_components=PCA_COMPONENTS,
    )
    X_reduced = projection_pca.fit_transform(X_scaled)

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=10,
        random_state=RANDOM_STATE,
    )
    kmeans_labels = kmeans.fit_predict(X_reduced)

    kmeans_ari = adjusted_rand_score(
        y,
        kmeans_labels,
    )

    dbscan_k = 4
    _, k_distances = plot_k_distance_curve(
        X_reduced,
        k=dbscan_k,
    )

    selected_eps = float(
        np.quantile(
            k_distances,
            0.90,
        )
    )

    dbscan = DBSCAN(
        eps=selected_eps,
        min_samples=dbscan_k + 1,
    )
    dbscan_labels = dbscan.fit_predict(X_reduced)

    dbscan_ari = adjusted_rand_score(
        y,
        dbscan_labels,
    )

    pca_projection_path = plot_pca_projection(
        X_reduced,
        y,
    )
    scree_path = plot_scree_curve(full_pca)

    (
        elbow_path,
        cluster_values,
        inertias,
    ) = plot_elbow_curve(X_reduced)

    noise_count = int(
        np.sum(dbscan_labels == DBSCAN.NOISE)
    )

    results: dict[str, Any] = {
        "dataset": "sklearn_digits",
        "n_samples": int(X.shape[0]),
        "n_original_features": int(X.shape[1]),
        "pca_components": PCA_COMPONENTS,
        "pca_explained_variance_ratio": (
            projection_pca.explained_variance_ratio_.tolist()
        ),
        "pca_cumulative_explained_variance": (
            projection_pca
            .cumulative_explained_variance()
            .tolist()
        ),
        "kmeans": {
            "n_clusters": N_CLUSTERS,
            "inertia": float(kmeans.inertia_),
            "iterations": int(kmeans.n_iter_),
            "adjusted_rand_index": float(kmeans_ari),
        },
        "dbscan": {
            "eps": selected_eps,
            "min_samples": dbscan.min_samples,
            "n_clusters": int(dbscan.n_clusters_),
            "noise_samples": noise_count,
            "adjusted_rand_index": float(dbscan_ari),
        },
        "elbow_curve": {
            "cluster_values": cluster_values.tolist(),
            "inertias": inertias.tolist(),
        },
        "figures": {
            "pca_projection": str(pca_projection_path),
            "pca_scree": str(scree_path),
            "kmeans_elbow": str(elbow_path),
            "dbscan_k_distance": str(
                FIGURES_DIR
                / "unsupervised_dbscan_k_distance.png"
            ),
        },
    }

    output_path = RESULTS_DIR / "unsupervised_analysis.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            results,
            output_file,
            indent=2,
        )

    return results


def main() -> dict[str, Any]:
    """Run Experiment 7 and print its main results."""
    results = run_experiment()

    print("Experiment 7 — Unsupervised Analysis")
    print(
        "K-Means ARI:",
        f"{results['kmeans']['adjusted_rand_index']:.4f}",
    )
    print(
        "DBSCAN ARI:",
        f"{results['dbscan']['adjusted_rand_index']:.4f}",
    )
    print(
        "DBSCAN noise samples:",
        results["dbscan"]["noise_samples"],
    )
    print(
        "Results saved to:",
        RESULTS_DIR / "unsupervised_analysis.json",
    )

    return results


if __name__ == "__main__":
    main()