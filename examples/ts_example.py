import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from core import TemporalSplitSampler, SeasonalSampler, SamplingStrategyFactory


def create_sample_time_series_data(n_series=10, n_timesteps=1000):
    """Создает пример данных временных рядов"""
    data = []
    for series_id in range(n_series):
        for t in range(n_timesteps):
            data.append({
                'series_id': series_id,
                'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(hours=t),
                'value': np.sin(0.1 * t + series_id) + 0.1 * np.random.randn(),
                'feature_1': np.random.randn(),
                'feature_2': np.random.randn()
            })
    return pd.DataFrame(data)


def demo_temporal_sampling():
    """Демонстрация временного семплирования"""
    print("=== Temporal Sampling Demo ===")

    # Создаем данные
    data = create_sample_time_series_data()
    print(f"Data shape: {data.shape}")

    # Временное разбиение
    temporal_sampler = TemporalSplitSampler(n_splits=4)
    temporal_sampler.fit(data, time_column='timestamp', series_id_column='series_id')
    partitions = temporal_sampler.get_partitions()

    print("Temporal partitions:")
    for partition_name, indices in partitions.items():
        print(f"  {partition_name}: {len(indices)} samples")

    # Сезонное семплирование
    seasonal_sampler = SeasonalSampler(seasonal_period=24)
    seasonal_sampler.fit(data, time_column='timestamp')
    seasonal_partitions = seasonal_sampler.get_partitions()

    print("\nSeasonal partitions:")
    for partition_name, indices in seasonal_partitions.items():
        print(f"  {partition_name}: {len(indices)} samples")


def demo_factory_pattern():
    """Демонстрация использования фабрики"""
    print("\n=== Factory Pattern Demo ===")

    data = create_sample_time_series_data()

    # Использование фабрики
    factory = SamplingStrategyFactory()
    strategy = factory.create_strategy('temporal_split', n_splits=3)

    strategy.fit(data, time_column='timestamp', series_id_column='series_id')
    partitions = strategy.get_partitions()

    print("Factory-created strategy partitions:")
    for partition_name, indices in partitions.items():
        print(f"  {partition_name}: {len(indices)} samples")


if __name__ == "__main__":
    demo_temporal_sampling()
    demo_factory_pattern()