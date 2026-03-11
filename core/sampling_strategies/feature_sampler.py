import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from .base_sampler import BaseSampler, HierarchicalStratifiedMixin
from ..repository.model_repo import SupportingModels
from ..utils.utils import to_dataframe, to_numpy

CLUSTERING_MODELS = SupportingModels.clustering_models.value
class FeatureBasedClusteringSampler(BaseSampler, HierarchicalStratifiedMixin):
    """
    Семплирование на основе кластеризации в пространстве признаков
    """

    def __init__(self, n_partitions: int = 5, method: str = 'kmeans',
                 feature_engineering: str = 'auto', random_state: int = 42, **kwargs):
        BaseSampler.__init__(self, random_state=random_state)
        HierarchicalStratifiedMixin.__init__(
            self,
            n_partitions=n_partitions,
            random_state=random_state,
            logger_name="FeatureBasedClusteringSampler",
        )
        self.n_clusters = n_partitions
        self.method = method
        self.feature_engineering = feature_engineering
        self.scaler = SupportingModels.scaling_models.value['scaler']()
        self.clusterer = None
        self.partitions = {}
        # Параметры для различных методов кластеризации
        self.clustering_params = kwargs

    def fit(self, data: Union[np.ndarray, pd.DataFrame],
            target: Optional[Union[pd.Series, np.ndarray]] = None, **kwargs) -> 'FeatureBasedClusteringSampler':
        """
        Args:
            data: Матрица признаков или сырые данные
            target: Целевая переменная (опционально)
        """
        # Извлечение признаков если необходимо
        data_df = to_dataframe(data)
        if self.feature_engineering == 'auto':
            features = self._auto_feature_engineering(data_df)
        else:
            features = data

        # Масштабирование признаков
        features = features.select_dtypes(include=[np.number])
        features = features.fillna(features.mean())
        features = to_numpy(features)

        features_scaled = self.scaler.fit_transform(features)

        # Применение dimensionality reduction если много признаков
        if features_scaled.shape[1] > 50:
            features_scaled = CLUSTERING_MODELS['pca'](n_components=50).fit_transform(features_scaled)

        # Кластеризация
        self.clusterer = CLUSTERING_MODELS[self.method](**self.clustering_params)
        cluster_labels = self.clusterer.fit_predict(features_scaled)

        # Создание разделов
        self.partitions = {}
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            if cluster_id != -1:  # Игнорируем шум для DBSCAN
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                self.partitions[f'chunk_{cluster_id}'] = cluster_indices

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

    def get_partitions(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Union[np.ndarray, pd.Series],
    ) -> Dict[Any, np.ndarray]:
        return self._get_partitions_default(data=data, target=target)


class TSNEClusteringSampler(FeatureBasedClusteringSampler):
    """
    Кластеризация на основе t-SNE проекции для визуализации и семплирования
    """

    def __init__(self, n_components: int = 2, perplexity: float = 30.0, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        self.n_components = n_components
        self.perplexity = perplexity
        self.tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=self.random_state)

    def fit(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'TSNEClusteringSampler':
        features_scaled = self.scaler.fit_transform(data)
        tsne_features = self.tsne.fit_transform(features_scaled)

        # Кластеризация в t-SNE пространстве
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = self.clusterer.fit_predict(tsne_features)

        self.partitions = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            self.partitions[f'tsne_cluster_{cluster_id}'] = cluster_indices

        return self
