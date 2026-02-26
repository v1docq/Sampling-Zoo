import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Literal
from collections import defaultdict
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
except Exception:  # pragma: no cover - optional dependency
    umap = None

from .base_sampler import BaseSampler
# from ..repository.model_repo import SupportingModels

class DelaunaySampler(BaseSampler):
    """
    Семплирование на основе триангуляции Делоне:
    делается кластеризация k-means, на центроидах кластеров строится триангуляция Делоне. 
    Для каждой точки из датасета определяется, к какому симплксу она относится, в соответствии с этим точки разбиваются на поднаборы.
    Если точка не попадает ни в один симплекс, она добавляяется в поднабор, соответствующий ближайшему к ней симплексу.

    Если симплексов в триангуляции окажется меньше, чем необходимо поднаборов, то некоторые (рандомно выбранные) симплексы продублируются.
    
    Если симплексов в триангуляции окажется больше, чем необходимо поднаборов, 
    то будет проведена кластеризация k-means на центроидах симплексов с числом кластеров равным требуемому числу поднаборов. 
    Далее симплексы из кластеров будут объединены.
     
    """
    def __init__(self, 
                 n_partitions: int = 10, 
                 random_state: int = 42, 
                 n_clusters: int = 5, 
                 emptiness_threshold: float = 0.01,
                 dim_reduction_method: Optional[Literal['pca', 'umap']] = None,
                 dim_reduction_target: int = 2
                 ):
        
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.n_partitions = n_partitions
        self.emptiness_threshold = emptiness_threshold
        self.n_clusters = n_clusters
        self.partitions = {}
        self.centroids = None
        self.triang = None
        self.kmeans_labels_ = None
        self.dim_reduction_method = dim_reduction_method
        self.dim_reduction_target = dim_reduction_target
        self.reducer = None

    def reduce_dimension(self, X: np.ndarray, fit_mode: bool = False) -> np.ndarray:
        """
        Универсальный метод для понижения размерности.
        Если fit_mode=True, инициализирует и обучает редьюсер.
        Если fit_mode=False, применяет уже обученный редьюсер.
        """
        # Если метод не задан, возвращаем исходные данные
        if self.dim_reduction_method is None:
            return X

        if self.dim_reduction_target >= X.shape[1]:
            return X

        # Логика обучения (fit)
        if fit_mode:
            if self.dim_reduction_method == 'pca':
                self.reducer = PCA(n_components=self.dim_reduction_target, 
                                   random_state=self.random_state)
            elif self.dim_reduction_method == 'umap':
                if umap is None:
                    raise ImportError(
                        "umap-learn is not installed. "
                        "Use dim_reduction_method='pca' or install umap-learn."
                    )
                self.reducer = umap.UMAP(n_components=self.dim_reduction_target, 
                                         random_state=self.random_state)
            else:
                raise ValueError(f"Unknown reduction method: {self.dim_reduction_method}")
            
            return self.reducer.fit_transform(X)
        
        # Логика применения (predict)
        else:
            if self.reducer is None:
                raise RuntimeError("Reducer has not been fitted. Call fit() first.")
            return self.reducer.transform(X)
                
    def fit(self, X: pd.DataFrame, y: pd.Series):

        # Преобразуем данные в numpy array, если это необходимо
        X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        # Понижение размерности, если необходимо
        X = self.reduce_dimension(X, fit_mode=True)

        # Проверка на случай, если n_clusters слишком мало для триангуляции
        if self.n_clusters < X.shape[1] + 1:
             print(f"Warning: For {X.shape[1]}-dimensional data, at least {X.shape[1] + 1} clusters are needed to form a simplex. "
                   f"Delaunay triangulation may fail.")
             
        temp_partitions = defaultdict(list)

        # Кластеризация для определения центроидов симплексов
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        kmeans.fit(X)
        self.centroids = kmeans.cluster_centers_
        self.triang = Delaunay(self.centroids)
        self.kmeans_labels_ = kmeans.labels_
        self.simplex_centers = self.centroids[self.triang.simplices].mean(axis=1)

        #Каждой точке сопоставим индекс одного из симплексов
        simplex_indices = self.predict_partitions(X)

        # Перегруппировка: каждому симплексу (=разделу) сопоставляются его точки
        for point_idx, simplex_idx in enumerate(simplex_indices):
            if simplex_idx != -1:
                temp_partitions[simplex_idx].append(point_idx)

        final_partitions = {idx: indices for idx, indices in temp_partitions.items()}

        # Удалим разделы, в которые попало мало точек (менее self.emptity_threshold от медианы по всем разделам)
        # Точки из этих разделов переместятся в ближайшие другие разделы
        final_partitions, self.kept_keys = self.delete_empty_simplexes(final_partitions, X_values=X)

        # Дублируем симплексы, если их оказалось меньше, чем n_partitions
        self.num_parts = len(final_partitions)
        if 0 < self.num_parts < self.n_partitions:
            final_partitions = self.duplicate_partitions(final_partitions)
        # Объединяем симплексы, если их оказалось больше, чем n_partitions
        if self.num_parts > self.n_partitions:
            final_partitions = self.merge_simplxes(final_partitions, self.kept_keys)

        # Формирование итогового словаря self.partitions
        sorted_keys = sorted(final_partitions.keys())[:self.n_partitions]
        self.partitions = {
            f"simplex_{key}": np.array(final_partitions[key])
            for key in sorted_keys
        }

        return self
    
    def predict_partitions(self, X: pd.DataFrame or np.ndarray) -> np.ndarray:
        """
        Определяет, к какому симплексу (разделу) относится каждая точка из X.

        Для каждой точки сначала определяется содержащий ее симплекс в триангуляции.
        Если точка находится вне выпуклой оболочки центроидов (и не попадает ни в один 
        симплекс), она относится к ближайшему симплексу по расстоянию до его центра.

        """
        
        # Преобразуем данные в numpy array, если это необходимо
        X_values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        # Понижение размерности, если необходимо
        X_values = self.reduce_dimension(X_values, fit_mode=False)

        # Вычисление, в какие симплексы попадают точки. 
        # Если точка не попадает ни в один симплекс, то пока -1
        simplex_indices = self.triang.find_simplex(X_values)
        outside_mask = (simplex_indices == -1)

        # Если такие точки есть, сопоставим им индекс ближайшего симплекса
        if np.any(outside_mask) and self.simplex_centers.shape[0] > 0:
            distances = cdist(X_values[outside_mask], self.simplex_centers)
            closest_simplex_indices = np.argmin(distances, axis=1)
            
            # np.argmin возвращает индексы в массиве self.simplex_centers.
            # Эти индексы напрямую соответствуют индексам симплексов в self.triang.simplices.
            simplex_indices[outside_mask] = closest_simplex_indices
        
        return simplex_indices

    def delete_empty_simplexes(self, partitions: dict, X_values: np.ndarray) -> dict:

        # Вычисляем размеры и порог на основе медианы
        sizes = {idx: len(points) for idx, points in partitions.items()}
        median_size = np.median(list(sizes.values()))
        threshold = median_size * self.emptiness_threshold

        # Определяем, какие симплексы удалить, а какие оставить
        to_delete_keys = {idx for idx, size in sizes.items() if size < threshold}
        to_keep_keys = list(set(partitions.keys()) - to_delete_keys)

        # Если удаляются все, ничего не делаем
        if not to_keep_keys:
            return partitions, list(partitions.keys())

        # Собираем точки из удаляемых симплексов
        points_to_reassign = []
        for key in to_delete_keys:
            points_to_reassign.extend(partitions[key])

        new_partitions = {key: partitions[key] for key in to_keep_keys}

        # Перераспределяем точки из удаленных симплексов
        if points_to_reassign:
            # Координаты центров оставленных симплексов
            kept_centers_indices = list(to_keep_keys)
            kept_centers_coords = self.simplex_centers[kept_centers_indices]
            
            # Находим ближайшие из оставшихся симплексов
            reassign_coords = X_values[points_to_reassign]
            distances = cdist(reassign_coords, kept_centers_coords)
            closest_kept_indices = np.argmin(distances, axis=1)

            # Распределяем точки по новым разделам
            for point_idx, new_local_idx in zip(points_to_reassign, closest_kept_indices):
                # Находим оригинальный индекс симплекса
                new_global_key = kept_centers_indices[new_local_idx]
                new_partitions[new_global_key].append(point_idx)

        return new_partitions, to_keep_keys

    def merge_simplxes(self, partitions_to_merge: dict, active_keys: list) -> dict:

        # Индексы и центроиды только для непустых симплексов
        original_indices = list(partitions_to_merge.keys())
        active_centroids = self.simplex_centers[active_keys]

        if len(active_keys) != len(partitions_to_merge):
         raise ValueError("Mismatch between partitions to merge and active keys.")

        # Кластеризуем центроиды симплексов, чтобы сгруппировать их
        kmeans_merger = KMeans(
            n_clusters=self.n_partitions, 
            n_init='auto', 
            random_state=self.random_state
        )
        labels = kmeans_merger.fit_predict(active_centroids)

        # Объединяем точки из старых симплексов в новые разделы
        merged_partitions = defaultdict(list)
        for original_idx, new_label in zip(original_indices, labels):
            merged_partitions[new_label].extend(partitions_to_merge[original_idx])

        return dict(merged_partitions)    

    def duplicate_partitions(self, partitions_to_duplicate: dict) -> dict:
        """
        Внутренний метод для дублирования симплексов путем случайного сэмплирования.
        """
        num_existing = len(partitions_to_duplicate)
        n_to_duplicate = self.n_partitions - num_existing
        
        available_keys = list(partitions_to_duplicate.keys())
        # Сэмплируем ключи симплексов, которые будем копировать
        keys_to_copy = self.rng.choice(available_keys, size=n_to_duplicate, replace=True)
        
        # Создаем копии
        next_new_key = (max(partitions_to_duplicate.keys()) + 1) if available_keys else 0
        for key in keys_to_copy:
            partitions_to_duplicate[next_new_key] = partitions_to_duplicate[key]
            next_new_key += 1
            
        return partitions_to_duplicate
    
    def get_partitions(self, data, target) -> Dict[Any, np.ndarray]:
        partition = {cluster: dict(feature=data.iloc[idx],
                                   target=target[idx]) for cluster, idx in self.partitions.items()}
        return partition
