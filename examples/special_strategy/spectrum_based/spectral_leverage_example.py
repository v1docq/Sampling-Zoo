import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Прямой импорт класса и через API фабрику
from core.sampling_strategies.spectral.spectral_leverage import SpectralLeverageSampler
from core.api.api_main import SamplingStrategyFactory


def generate_low_rank_matrix(n_samples=1000, n_features=50, latent_rank=5, noise_level=0.1, seed=42):
    rng = np.random.default_rng(seed)
    U_true = rng.standard_normal((n_samples, latent_rank))
    V_true = rng.standard_normal((latent_rank, n_features))
    X = U_true @ V_true + noise_level * rng.standard_normal((n_samples, n_features))

    # Добавим "выброс" с большой нормой, который должен иметь высокий leverage score
    X[0, :] = 50.0 * rng.standard_normal(n_features)
    return X


def main():
    X = generate_low_rank_matrix(n_samples=1000, n_features=50, latent_rank=5, noise_level=0.1)
    print(f"Размер данных: {X.shape}")

    # 1) Прямой подход: использовать класс напрямую
    sampler = SpectralLeverageSampler(
        sample_size=10,
        approx_rank=5,
        random_state=42,
        return_weights=True,
    )

    sampler.fit(X)

    res = sampler.sample_indices()
    if isinstance(res, tuple):
        indices, weights = res
    else:
        indices = res
        weights = None

    print("\n--- Результаты (прямой вызов) ---")
    print(f"Выбранные индексы: {indices}")
    if weights is not None:
        print(f"Веса (importance weights): {np.round(weights, 3)}")

    print(f"\nLeverage score для индекса 0 (выброс): {sampler.sampling_scores_[0]:.6f}")
    print(f"Средний leverage score: {np.mean(sampler.sampling_scores_):.6f}")

    if 0 in indices:
        print(">> Успех: Выброс (индекс 0) был выбран, так как он важен для структуры.")
    else:
        print(">> Выброс не попал в выборку (это возможно, но менее вероятно).")

    assert len(indices) == 10
    assert len(set(indices)) == 10, "Индексы должны быть уникальны (replace=False)"

    # Визуализация распределения leverage-скоров (топ-50)
    top_k = 50
    scores = sampler.sampling_scores_
    top_idx = np.argsort(scores)[-top_k:][::-1]

    plt.figure(figsize=(10, 4))
    plt.bar(range(top_k), scores[top_idx], color='C0')
    # Отметим выброс (индекс 0), если он попал в топ
    if 0 in top_idx:
        pos = list(top_idx).index(0)
        plt.bar(pos, scores[0], color='C3', label='Outlier (index 0)')
    plt.title('Top leverage scores (Top {})'.format(top_k))
    plt.xlabel('Rank (by leverage score)')
    plt.ylabel('Leverage score')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Через API фабрику
    factory = SamplingStrategyFactory()
    strategy_kwargs = dict(sample_size=10, approx_rank=5, random_state=42, return_weights=True)

    sampler_api = factory.create_strategy('spectral_leverage', **strategy_kwargs)
    sampler_api.fit(X)

    res_api = sampler_api.sample_indices()
    if isinstance(res_api, tuple):
        indices_api, weights_api = res_api
    else:
        indices_api = res_api
        weights_api = None

    print("\n--- Результаты (через API) ---")
    print(f"Выбранные индексы: {indices_api}")
    if weights_api is not None:
        print(f"Веса (importance weights): {np.round(weights_api, 3)}")

    print(f"Leverage score для индекса 0 (выброс): {sampler_api.sampling_scores_[0]:.6f}")

    if 0 in indices_api:
        print(">> Успех: Выброс (индекс 0) был выбран.")
    else:
        print(">> Выброс не попал в выборку.")


if __name__ == '__main__':
    main()
