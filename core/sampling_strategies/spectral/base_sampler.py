from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from ..base_sampler import BaseSampler


class SpectralSamplerBase(BaseSampler):
    def __init__(
        self,
        sample_size: int,
        approx_rank: int | float  = 1.0,
        random_state: int | None = None,
        return_weights: bool = False,
        backend_config: dict | None = None,
    ):
        self.sample_size = sample_size
        self.approx_rank = approx_rank
        self.random_state = random_state
        self.return_weights = return_weights
        self.backend_config = backend_config

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SpectralSamplerBase":
        """Обучает семплер на данных X (и y, если предоставлено)"""
        pass
    
    @abstractmethod
    def build_spectral_representation(self, X):
        """Строит спектральное представление данных X - аппроксимацию пространства на основе первых approx_rank собственных векторов."""
        pass

    @abstractmethod
    def compute_sampling_scores(self):
        """Вычисляет оценки для семплирования на основе спектрального представления."""
        pass
    
    @abstractmethod
    def sample_indices(self, replace: bool = False) -> List[int]:
        """
        Семплирует выборку размера sample_size из индексов на основе вычисленных оценок семплирования.
        Возвращает также веса для выбранных образцов, если в return_weights установлено True.
        Семплирование может быть с возвращением или без, в зависимости от параметра replace.
        """
        pass

    @abstractmethod
    def get_partitions(self) -> Dict[Any, np.ndarray]:
        """
        Разделяет данные на разделы и возвращает индексы для каждого раздела.  
        """
        pass

