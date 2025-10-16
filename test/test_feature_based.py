import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import FeatureBasedClusteringSampler, TSNEClusteringSampler


class TestFeatureBasedSamplers(unittest.TestCase):

    def setUp(self):
        """Создает тестовые табличные данные"""
        np.random.seed(42)
        self.X = np.random.randn(200, 10)
        self.y = np.random.randint(0, 2, 200)
        self.df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])

    def test_feature_clustering_initialization(self):
        """Тест инициализации FeatureBasedClusteringSampler"""
        sampler = FeatureBasedClusteringSampler(n_clusters=3, method='kmeans')
        self.assertEqual(sampler.n_clusters, 3)
        self.assertEqual(sampler.method, 'kmeans')

    def test_feature_clustering_fit(self):
        """Тест обучения FeatureBasedClusteringSampler"""
        sampler = FeatureBasedClusteringSampler(n_clusters=3)
        sampler.fit(self.df, target=self.y)

        partitions = sampler.get_partitions()

        # Проверяем что создалось правильное количество кластеров
        self.assertEqual(len(partitions), 3)

        # Проверяем что все данные распределены по кластерам
        total_samples = sum(len(indices) for indices in partitions.values())
        self.assertEqual(total_samples, len(self.df))

    def test_feature_clustering_with_numpy(self):
        """Тест работы с numpy массивами"""
        sampler = FeatureBasedClusteringSampler(n_clusters=2)
        sampler.fit(self.X, target=self.y)

        partitions = sampler.get_partitions()
        self.assertEqual(len(partitions), 2)

    def test_tsne_clustering(self):
        """Тест TSNEClusteringSampler"""
        sampler = TSNEClusteringSampler(n_clusters=2, n_components=2)
        sampler.fit(self.X, target=self.y)

        partitions = sampler.get_partitions()
        self.assertEqual(len(partitions), 2)


if __name__ == '__main__':
    unittest.main()