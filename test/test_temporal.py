import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import TemporalSplitSampler, SeasonalSampler


class TestTemporalSamplers(unittest.TestCase):

    def setUp(self):
        """Создает тестовые данные"""
        # Создаем временные ряды
        self.time_series_data = pd.DataFrame({
            'series_id': [1] * 100 + [2] * 100,
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H').tolist() * 2,
            'value': np.random.randn(200),
            'feature_1': np.random.randn(200)
        })

    def test_temporal_split_sampler_initialization(self):
        """Тест инициализации TemporalSplitSampler"""
        sampler = TemporalSplitSampler(n_splits=4)
        self.assertEqual(sampler.n_splits, 4)
        self.assertEqual(sampler.method, 'sequential')

    def test_temporal_split_fit(self):
        """Тест обучения TemporalSplitSampler"""
        sampler = TemporalSplitSampler(n_splits=4)
        sampler.fit(self.time_series_data, time_column='timestamp', series_id_column='series_id')

        partitions = sampler.get_partitions()

        # Проверяем что создалось правильное количество разделов
        self.assertEqual(len(partitions), 4)

        # Проверяем что все индексы уникальны
        all_indices = []
        for indices in partitions.values():
            all_indices.extend(indices)

        self.assertEqual(len(all_indices), len(set(all_indices)))
        self.assertEqual(len(all_indices), len(self.time_series_data))

    def test_temporal_split_transform(self):
        """Тест преобразования данных TemporalSplitSampler"""
        sampler = TemporalSplitSampler(n_splits=4)
        sampler.fit(self.time_series_data, time_column='timestamp', series_id_column='series_id')

        transformed_data = sampler.transform(self.time_series_data)

        # Проверяем что возвращается словарь с DataFrame
        self.assertIsInstance(transformed_data, dict)
        for partition_data in transformed_data.values():
            self.assertIsInstance(partition_data, pd.DataFrame)

    def test_seasonal_sampler(self):
        """Тест SeasonalSampler"""
        sampler = SeasonalSampler(seasonal_period=24)
        sampler.fit(self.time_series_data, time_column='timestamp')

        partitions = sampler.get_partitions()

        # Проверяем что создались разделы
        self.assertGreater(len(partitions), 0)

        # Проверяем что все данные распределены
        total_samples = sum(len(indices) for indices in partitions.values())
        self.assertEqual(total_samples, len(self.time_series_data))


if __name__ == '__main__':
    unittest.main()