from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..base_sampler import BaseSampler


class SpectralSamplerBase(BaseSampler):
    """Base class for samplers driven by spectral or tensor representations."""

    def __init__(
        self,
        sample_size: Optional[int] = None,
        approx_rank: Union[int, float] = 1.0,
        random_state: Union[int, None] = None,
        return_weights: bool = False,
        backend_config: Union[dict, None] = None,
        n_partitions: int = 1,
        n_views: int = 1,
        view_strategy: str = "gaussian",
        chunk_fraction: float = 1.0,
        chunks_percent: float = 100.0,
        min_chunk_size: int = 1,
        max_chunk_size: Optional[int] = None,
        selection_method: str = "all",
        leverage_cap_quantile: float = 0.95,
        routing_temperature: float = 1.0,
        routing_shrinkage: float = 0.0,
        backend: str = "auto",
        device: str = "cpu",
        dtype: str = "float32",
        include_categorical: bool = True,
        max_one_hot_cardinality: int = 128,
        max_encoded_features: Optional[int] = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(random_state=42 if random_state is None else random_state, **kwargs)
        self.sample_size = sample_size
        self.approx_rank = approx_rank
        self.return_weights = return_weights
        self.backend_config = backend_config

        self.n_partitions = self._validate_positive_int("n_partitions", n_partitions)
        self.n_views = self._validate_positive_int("n_views", n_views)
        self.view_strategy = self._validate_choice("view_strategy", view_strategy, ("subsample", "gaussian"))
        self.chunk_fraction = self._validate_fraction("chunk_fraction", chunk_fraction)
        self.chunks_percent = self._validate_percent("chunks_percent", chunks_percent)
        self.min_chunk_size = self._validate_positive_int("min_chunk_size", min_chunk_size)
        self.max_chunk_size = None if max_chunk_size is None else self._validate_positive_int(
            "max_chunk_size",
            max_chunk_size,
        )
        self.selection_method = self._validate_choice(
            "selection_method",
            selection_method,
            ("all", "leverage", "capped_leverage", "maxvol", "hybrid"),
        )
        self.leverage_cap_quantile = self._validate_fraction(
            "leverage_cap_quantile",
            leverage_cap_quantile,
        )
        self.routing_temperature = float(routing_temperature)
        self.routing_shrinkage = float(routing_shrinkage)
        self.backend = self._validate_choice("backend", backend, ("auto", "torch", "numpy"))
        self.backend_: Optional[str] = None
        self.device = device
        self.dtype = self._validate_choice("dtype", dtype, ("float32", "float64"))
        self.show_progress = bool(show_progress)
        self._configure_tabular_preprocessing(
            include_categorical=include_categorical,
            max_one_hot_cardinality=max_one_hot_cardinality,
            max_encoded_features=max_encoded_features,
        )
        self._init_spectral_state()

    def _init_spectral_state(self) -> None:
        self.view_specs_: List[Any] = []
        self.singular_values_: Optional[np.ndarray] = None
        self.right_basis_: Optional[np.ndarray] = None
        self.sample_embedding_: Optional[np.ndarray] = None
        self.leverage_scores_: Optional[np.ndarray] = None
        self.clusterer_: Any = None
        self.cluster_labels_: Optional[np.ndarray] = None
        self.partition_names_: List[str] = []
        self.partition_to_cluster_: Dict[str, int] = {}
        self.partitions: Dict[str, np.ndarray] = {}
        self.diagnostics_: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y=None) -> "SpectralSamplerBase":
        """Fit the sampler on data."""
        raise NotImplementedError

    def build_spectral_representation(self, X: np.ndarray) -> Any:
        """Build a spectral representation for samplers that expose this step."""
        raise NotImplementedError

    def compute_sampling_scores(self) -> Any:
        """Compute sampling scores for samplers that expose this step."""
        raise NotImplementedError

    def sample_indices(self, replace: bool = False) -> List[int]:
        """Sample indices from fitted spectral scores."""
        raise NotImplementedError

    def get_partitions(self) -> Dict[Any, np.ndarray]:
        """Return fitted partitions."""
        raise NotImplementedError
