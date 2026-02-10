import numpy as np
import pandas as pd
from .base_sampler import SpectralSamplerBase
from .backend.matrix_backend import RandomizedSVDBackend

class SpectralLeverageSampler(SpectralSamplerBase): 
    def __init__(
        self,
        sample_size: int,
        approx_rank: int | float = 1.0,
        random_state: int | None = None,
        return_weights: bool = False,
        backend_config: dict | None = None,
    ):
        super().__init__(
            sample_size=sample_size,
            approx_rank=approx_rank,
            random_state=random_state,
            return_weights=return_weights,
            backend_config=backend_config,
        )

    def fit(self, X: np.ndarray | pd.DataFrame | None = None) -> "SpectralLeverageSampler":
        """Обучает семплер на данных X (и y, если предоставлено)"""
    
        # Преобразуем данные в numpy array, если это необходимо
        X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    
        self.build_spectral_representation(X)
        self.compute_sampling_scores()
        return self 
    
    def build_spectral_representation(self, X):
        """
        Строит спектральное представление данных X - аппроксимацию пространства на основе первых approx_rank собственных векторов.
        Пока бэкенд реализован только с использованием randomized SVD из sklearn. И для него используются параметры oversample_factor и power_iterations."""
        
        backend = RandomizedSVDBackend(
            oversample_factor=self.backend_config.get('oversample_factor', 10) if self.backend_config else 10,
            power_iterations=self.backend_config.get('power_iterations', 2) if self.backend_config else 2,
            random_state=self.random_state
        )
        self.spectral_basis_, _, _ = backend.compute_basis(X, rank=self.approx_rank)

    def compute_sampling_scores(self):
        leverage_scores = np.sum(self.spectral_basis_ ** 2, axis=1)
        self.sampling_scores_ = leverage_scores / np.sum(leverage_scores)

    def sample_indices(self, replace: bool = False) -> list[int]:
        rng = np.random.default_rng(self.random_state)
        sampled_indices = rng.choice(
            len(self.sampling_scores_),
            size=self.sample_size,
            replace=replace,
            p=self.sampling_scores_
        )
        if self.return_weights:
            weights = 1.0 / (self.sampling_scores_[sampled_indices] + 1e-12)
            return sampled_indices.tolist(), weights
        return sampled_indices.tolist()

    def get_partitions(self) -> dict:
        pass
        