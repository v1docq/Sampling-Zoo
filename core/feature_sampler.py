import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, Any, Union
from .base_sampler import BaseSampler


class FeatureBasedClusteringSampler(BaseSampler):
    """
    Семплирование на основе кластеризации в пространстве признаков
    """

    def __init__(self, n_clusters: int = 5, method: str = 'kmeans',
                 feature_engineering: str = 'auto', **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.method = method
        self.feature_engineering = feature_engineering
        self.scaler = StandardScaler()
        self.clusterer = None

        # Параметры для различных методов кластеризации
        self.clustering_params = kwargs

    def fit(self, data: Union[np.ndarray, pd.DataFrame],
            target: np.ndarray = None, **kwargs) -> 'FeatureBasedClusteringSampler':
        """
        Args:
            data: Матрица признаков или сырые данные
            target: Целевая переменная (опционально)
        """
        # Извлечение признаков если необходимо
        if self.feature_engineering == 'auto' and isinstance(data, pd.DataFrame):
            features = self._auto_feature_engineering(data)
        else:
            features = data

        # Масштабирование признаков
        if isinstance(features, pd.DataFrame):
            features = features.select_dtypes(include=[np.number])
            features = features.fillna(features.mean())

        features_scaled = self.scaler.fit_transform(features)

        # Применение dimensionality reduction если много признаков
        if features_scaled.shape[1] > 50:
            features_scaled = PCA(n_components=50).fit_transform(features_scaled)

        # Кластеризация
        if self.method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **self.clustering_params
            )
        elif self.method == 'dbscan':
            self.clusterer = DBSCAN(**self.clustering_params)
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

        cluster_labels = self.clusterer.fit_predict(features_scaled)

        # Создание разделов
        self.partitions_ = {}
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            if cluster_id != -1:  # Игнорируем шум для DBSCAN
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                self.partitions_[f'cluster_{cluster_id}'] = cluster_indices

        return self

    def _auto_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Автоматическое извлечение признаков из DataFrame"""
        numeric_data = data.select_dtypes(include=[np.number])

        # Добавляем статистические признаки если мало колонок
        if len(numeric_data.columns) < 10:
            statistical_features = []
            for col in numeric_data.columns:
                statistical_features.extend([
                    f'{col}_mean', f'{col}_std', f'{col}_skew', f'{col}_kurtosis'
                ])
            # Здесь можно добавить реальное вычисление статистик
            # для демонстрации возвращаем исходные данные
            return numeric_data

        return numeric_data

    def get_partitions(self) -> Dict[Any, np.ndarray]:
        return self.partitions_


class TSNEClusteringSampler(FeatureBasedClusteringSampler):
    """
    Кластеризация на основе t-SNE проекции для визуализации и семплирования
    """

    def __init__(self, n_components: int = 2, perplexity: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.perplexity = perplexity
        self.tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=self.random_state)

    def fit(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'TSNEClusteringSampler':
        features_scaled = self.scaler.fit_transform(data)
        tsne_features = self.tsne.fit_transform(features_scaled)

        # Кластеризация в t-SNE пространстве
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = self.clusterer.fit_predict(tsne_features)

        self.partitions_ = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            self.partitions_[f'tsne_cluster_{cluster_id}'] = cluster_indices

        return self