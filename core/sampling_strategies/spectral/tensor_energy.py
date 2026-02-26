from typing import List, Union, Tuple
# from .backend.tensor_backend import HOSVDBackend
from .backend.matrix_backend import RandomizedSVDBackend
from .base_sampler import SpectralSamplerBase
import numpy as np
import pandas as pd
try:
    import tensorly as tl
except Exception:  # pragma: no cover - optional dependency
    tl = None


def _unfold_tensor(x: np.ndarray, mode: int) -> np.ndarray:
    if tl is not None:
        return tl.unfold(x, mode)
    moved = np.moveaxis(x, mode, 0)
    return moved.reshape(moved.shape[0], -1)

class TensorEnergySampler(SpectralSamplerBase):
    """
    Семплер, использующий спектральную энергию тензора для выбора наиболее информативных подтензоров.
    
    Работает через вычисление leverage scores для развёрток тензора вдоль заданных мод
    и последующее семплирование из совместного распределения вероятностей.
    """
    def __init__(
        self,
        sample_size: int,
        modes: List[int], # Список мод, по которым проводится семплирование
        approx_rank: Union[int, float, List[Union[int, float]]] = 1.0, # Может быть общим или списком для каждой моды
        random_state: Union[int, None] = None,
        return_weights: bool = False,
        backend_config: Union[dict, None] = None,
    ):
        super().__init__(sample_size, approx_rank, random_state, return_weights, backend_config)
        self.modes = modes
        self.backend = RandomizedSVDBackend(
            random_state=random_state 
        )
        
        # Обработка рангов: если передано одно число, используем его для всех мод
        if isinstance(approx_rank, list):
            if len(approx_rank) != len(modes):
                raise ValueError("Длина списка approx_rank должна совпадать с количеством мод.")
            self.ranks_per_mode = approx_rank
        else:
            self.ranks_per_mode = [approx_rank] * len(modes)

        # Хранилище для вычисленных вероятностей и структур
        self.factor_matrices = {} # Словарь {mode: U_matrix}
        self.mode_probs = {}      # Словарь {mode: probability_vector}
        self.joint_probs = None   # Совместное распределение (тензор)
        self.sampled_indices_ = None # Результат семплирования
        self.weights_ = None         # Веса

    def fit(self, X = None) -> "TensorEnergySampler":
        """
        Выполняет полный цикл подготовки: построение спектрального представления и вычисление scores.
        """

        # Преобразуем данные в numpy array, если это необходимо
        X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        # Валидация входных данных
        if X.ndim <= max(self.modes):
            raise ValueError(f"Тензор имеет размерность {X.ndim}, но запрошена мода {max(self.modes)}")
            
        self.build_spectral_representation(X)
        self.compute_sampling_scores()
        return self

    def build_spectral_representation(self, X: np.ndarray):
        """
        Для каждой целевой моды разворачивает тензор и вычисляет SVD для получения левых сингулярных векторов U.
        """
        for i, mode in enumerate(self.modes):
            X_unfolded = _unfold_tensor(X, mode)
            rank = self.ranks_per_mode[i]
            U, _, _ = self.backend.compute_basis(X_unfolded, rank=rank)
            self.factor_matrices[mode] = U

    def compute_sampling_scores(self):
        """
        Вычисляет leverage scores для каждой моды и строит совместное распределение вероятностей.
        """
        # Вычисление вероятностей для каждой моды: leverage scores + нормализация
        for mode in self.modes:
            U = self.factor_matrices[mode]
            scores = np.sum(U**2, axis=1)
            probs = scores / np.sum(scores)
            self.mode_probs[mode] = probs

        # Построение совместного распределения
        # joint_p(i, j, k) = p(i) * p(j) * p(k)
        
        # Инициализация с вероятностями первой моды
        self.joint_probs = self.mode_probs[self.modes[0]]

        # Итеративно добавляем остальные моды через внешнее произведение (outer product)
        for mode in self.modes[1:]:
            current_probs = self.mode_probs[mode]
            # np.multiply.outer создает тензор с размерностью dim(prev) + 1
            self.joint_probs = np.multiply.outer(self.joint_probs, current_probs)
            
        # В итоге self.joint_probs имеет размерность len(self.modes)
        # и форму (n_mode1, n_mode2, ...)
        # То, что мы взяли внешнее произведение от распределений, гарантирует, что сумма по всем элементам равна 1

    def sample_indices(self, replace: bool = False) -> Union[List[Tuple[int, ...]], Tuple[List[Tuple[int, ...]], np.ndarray]]:
        """
        Семплирует мульти-индексы из совместного распределения.
        """
        if self.joint_probs is None:
            raise RuntimeError("Сначала нужно вызвать fit()")

        # Flatten совместного распределения для np.random.choice
        flat_probs = self.joint_probs.flatten()
        
        # На всякий случай нормализация)
        flat_probs /= flat_probs.sum()
        
        n_total_options = flat_probs.shape[0]
        
        flat_indices = np.random.choice(
            n_total_options,
            size=self.sample_size,
            p=flat_probs,
            replace=replace
        )
        
        # Преобразование обратно в мульти-индексы
        # unravel_index возвращает кортеж массивов (arr_dim1, arr_dim2, ...)
        multi_indices_arrays = np.unravel_index(flat_indices, self.joint_probs.shape)
        
        # Преобразуем в список кортежей [(i1, j1), (i2, j2), ...] с координатами каждого выбранного подтензора
        # zip(*arrays) транспонирует структуру
        self.sampled_indices = list(zip(*multi_indices_arrays))
        
        if self.return_weights:
            selected_probs = flat_probs[flat_indices]
            self.weights = 1.0 / (selected_probs + 1e-12)
            return self.sampled_indices, self.weights
            
        return self.sampled_indices
    
    def get_partitions(self) -> dict:
        """
        Разделяет данные на разделы и возвращает индексы для каждого раздела.  
        Пока не реализовано для тензоров.
        """
        pass
