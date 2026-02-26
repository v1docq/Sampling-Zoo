import numpy as np
import pandas as pd
try:
    import hdbscan
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None
from typing import Dict, Any, Union
from collections import defaultdict
from sklearn.cluster import KMeans

from .base_sampler import BaseSampler
# from ..repository.model_repo import SupportingModels

class HDBScanSampler(BaseSampler):
    """
    Семплирование на основе кластеризации HDBSCAN.
    Для каждой точки определяется вероятность принадлежности к каждому кластеру.
    
    Если one_cluster == Truе, то каждая точка включается в кластер с наибольшей вероятностью. 
    Если one_cluster == False, то точка включается в кластеры с вероятностями, превышающими prob_threshold.

    Если all_points = True, то точки, которые hdbscan отнес к шуму, будут распределены по кластерам.
    Если all_points = False, то точки шум не будет включен в итоговые классы.
    """
    def __init__(self, 
                 min_cluster_size: int = 10, 
                 one_cluster: bool = True, 
                 prob_threshold: float = None, 
                 all_points = True, 
                 random_state: int = 42
                 ):
        
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size, prediction_data=True) if hdbscan is not None else None
        self._dbscan_fallback = hdbscan is None
        self.partitions = {}
        self.prob_threshold = prob_threshold
        self.one_cluster = one_cluster
        self.all_points = all_points
        if one_cluster == False and prob_threshold == None:
            raise ValueError("Для режима one_cluster=False необходимо задать prob_threshold")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        
        # Преобразуем данные в numpy array, если это необходимо
        X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        temp_partitions = defaultdict(list)

        if self._dbscan_fallback:
            # Fallback for environments without hdbscan package.
            # Use KMeans to provide partitioning-compatible behavior with linear runtime.
            n_clusters = max(2, min(20, X.shape[0] // max(1, self.min_cluster_size)))
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
            labels = self.clusterer.fit_predict(X)
            for i, label in enumerate(labels):
                temp_partitions[int(label)].append(i)

            self.partitions = {
                f"cluster_{label}": np.array(indices)
                for label, indices in temp_partitions.items()
            }
            return self

        self.clusterer.fit(X)

        membership_vectors = hdbscan.all_points_membership_vectors(self.clusterer)
        
        n_samples = X.shape[0]
        
        if self.one_cluster:
            # Каждая точка, включая шум, отправляется в наиболее вероятный кластер
            best_clusters = np.argmax(membership_vectors, axis=1)
            for i, cluster_id in enumerate(best_clusters):
                temp_partitions[cluster_id].append(i)
        else:
            for i in range(n_samples):
                probs = membership_vectors[i]
                passing_clusters = np.where(probs >= self.prob_threshold)[0]
                
                if passing_clusters.size > 0:
                    # Если есть уверенные кандидаты — берем их
                    for cluster_label in passing_clusters:
                        temp_partitions[cluster_label].append(i)
                elif self.all_points == True:
                    # Если нет и all_points = True, берем наилучший по вероятности
                    best_cluster_label = np.argmax(probs)
                    temp_partitions[best_cluster_label].append(i)

        self.partitions = {
            f"cluster_{label}": np.array(indices)
            for label, indices in temp_partitions.items()
        }
        
        return self
    
    def predict_partitions(self, X: pd.DataFrame or np.ndarray):
        """
        Определяет принадлежность новых точек к кластерам.

        Args:
            X: Матрица признаков (N_samples, N_features).

        Returns:
            Если one_cluster=True: 
                np.ndarray shape (N_samples,) с ID кластера для каждой точки.
            Если one_cluster=False: 
                List[np.ndarray] длиной N_samples, где каждый элемент — массив ID кластеров.
        """

        X_values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        
        # Защита от подачи одномерного массива (одной точки)
        X_values = np.atleast_2d(X_values)

        if self._dbscan_fallback:
            labels = self.clusterer.predict(X_values)
            if self.one_cluster:
                return labels
            return [np.array([int(label)]) for label in labels]

        prob_matrix = hdbscan.membership_vector(self.clusterer, X_values)

        if self.one_cluster:
            return np.argmax(prob_matrix, axis=1)
        
        else:
            # Возвращаем список массивов индексов
            result = []
            n_samples = prob_matrix.shape[0]
            
            for i in range(n_samples):
                probs = prob_matrix[i]
                passing_clusters = np.where(probs >= self.prob_threshold)[0]
                
                # если точка не попала ни в один кластер, берем максимальную вероятность
                if passing_clusters.size == 0:
                    passing_clusters = np.array([np.argmax(probs)])
                
                result.append(passing_clusters)
            
            return result

    def get_partitions(self, data, target) -> Dict[Any, np.ndarray]:
        partition = {cluster: dict(feature=data.iloc[idx],
                                   target=target[idx]) for cluster, idx in self.partitions.items()}
        return partition
