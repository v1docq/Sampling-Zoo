import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from core import (FeatureBasedClusteringSampler, DifficultyBasedSampler,
                  UncertaintySampler, SamplingStrategyFactory)


def create_sample_tabular_data(n_samples=1000, n_features=20, problem_type='classification'):
    """Создает пример табличных данных"""
    if problem_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=10, n_redundant=5, random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features,
            n_informative=10, random_state=42
        )

    feature_names = [f'feature_{i}' for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names), y


def demo_feature_based_sampling():
    """Демонстрация feature-based семплирования"""
    print("=== Feature-Based Sampling Demo ===")

    # Создаем данные
    X, y = create_sample_tabular_data(n_samples=2000, problem_type='classification')
    print(f"Data shape: {X.shape}")

    # Кластеризация в пространстве признаков
    cluster_sampler = FeatureBasedClusteringSampler(n_clusters=4, method='kmeans')
    cluster_sampler.fit(X, target=y)
    partitions = cluster_sampler.get_partitions()

    print("Feature-based partitions:")
    for partition_name, indices in partitions.items():
        print(f"  {partition_name}: {len(indices)} samples")

    # t-SNE кластеризация
    tsne_sampler = FeatureBasedClusteringSampler(n_clusters=3, method='kmeans')
    tsne_sampler.fit(X, target=y)
    tsne_partitions = tsne_sampler.get_partitions()

    print("\nt-SNE partitions:")
    for partition_name, indices in tsne_partitions.items():
        print(f"  {partition_name}: {len(indices)} samples")


def demo_difficulty_based_sampling():
    """Демонстрация difficulty-based семплирования"""
    print("\n=== Difficulty-Based Sampling Demo ===")

    X, y = create_sample_tabular_data(n_samples=1500, problem_type='classification')

    # Семплирование по сложности
    difficulty_sampler = DifficultyBasedSampler(difficulty_threshold=0.7)
    difficulty_sampler.fit(X, y)
    partitions = difficulty_sampler.get_partitions()
    difficulty_scores = difficulty_sampler.get_difficulty_scores()

    print("Difficulty-based partitions:")
    for partition_name, indices in partitions.items():
        print(f"  {partition_name}: {len(indices)} samples")

    print(f"Difficulty scores range: [{difficulty_scores.min():.3f}, {difficulty_scores.max():.3f}]")

    # Uncertainty sampling
    uncertainty_sampler = UncertaintySampler(uncertainty_threshold=0.6)
    uncertainty_sampler.fit(X, y)
    uncertainty_partitions = uncertainty_sampler.get_partitions()
    uncertainty_scores = uncertainty_sampler.get_uncertainty_scores()

    print("\nUncertainty partitions:")
    for partition_name, indices in uncertainty_partitions.items():
        print(f"  {partition_name}: {len(indices)} samples")

    print(f"Uncertainty scores range: [{uncertainty_scores.min():.3f}, {uncertainty_scores.max():.3f}]")


if __name__ == "__main__":
    demo_feature_based_sampling()
    demo_difficulty_based_sampling()