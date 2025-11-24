import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from collections import defaultdict
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from .base_sampler import BaseSampler

class VoronoiSampler(BaseSampler):
    '''
    Класс для семплирования на основе решетки Вороного.
    Делается кластеризация k-means, на центроидах кластеров строится решетка Вороного.
    Для каждой точки из датасета определяется, к какой ячейке она относится, в соответствии с этим точки разбиваются на поднаборы.

    '''
    def __init__(self, n_partitions: int = 10, random_state: int = 42, emptiness_threshold: float = 0.01):
        
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.n_partitions = n_partitions
        self.emptiness_threshold = emptiness_threshold
        self.n_clusters = n_partitions
        self.partitions = {}
        self.centroids = None
        self.kmeans_labels_ = None
        self.kmeans = KMeans(
            n_clusters=n_partitions, 
            random_state=random_state, 
            n_init='auto'
        )

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        # Преобразуем X, сохраняя возможность работы с numpy
        X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        # Обучение KMeans. labels_ уже содержит принадлежность к ячейкам Вороного
        self.kmeans.fit(X)
        self.centroids = self.kmeans.cluster_centers_
        
        labels = self.kmeans.labels_
        
        # Группируем индексы по кластерам
        self.partitions = {f"cluster_{k}": [] for k in range(self.n_partitions)}
        for idx, label in enumerate(labels):
            self.partitions[f"cluster_{label}"].append(idx)
            
        self.partitions = {f"cluster_{k}": np.array(v) for k, v in self.partitions.items()}

        return self
    
    def predict_partitions(self, X: pd.DataFrame or np.ndarray):
        """
        Определяет принадлежность новых точек к кластерам.

        Args:
            X: Матрица признаков (N_samples, N_features).

        Returns:
            result: np.ndarray длиной N_samples, где каждый элемент - ID кластера для точки.
        """
        # Преобразуем данные в numpy array, если это необходимо
        X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        # Используем оптимизированный метод predict из sklearn
        return self.kmeans.predict(X)
        
    def get_partitions(self, data, target) -> Dict[Any, np.ndarray]:
        partition = {cluster: dict(feature=data.iloc[idx],
                                   target=target[idx]) for cluster, idx in self.partitions.items()}
        return partition
