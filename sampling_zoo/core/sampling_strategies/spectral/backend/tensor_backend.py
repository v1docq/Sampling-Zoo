import numpy as np
import torch 
import tensorly

class HOSVDBackend:
    def __init__(self, random_state=None):
        self.random_state = random_state

    def compute_hosvd(self, X: np.ndarray, rank: list[int] | float, **kwargs):
        """
        Вычисляет HOSVD тензора X с заданными рангами по каждому измерению. Пока использует tensorly с бэкендом PyTorch.
        Arguments:  
        X: входной тензор
        rank: список рангов для каждого измерения тензора или float в (0,1], обозначающий долю от размера соответствующего измерения
        **kwargs: дополнительные параметры для HOSVD
        Returns:
        core_tensor: core tensor after HOSVD
        factors: list of factor matrices for each mode
        """
        tensorly.set_backend('pytorch')
        torch.manual_seed(self.random_state if self.random_state is not None else 0)

        if isinstance(rank, float):
            rank = [int(rank * dim) for dim in X.shape]
        X_tensor = tensorly.tensor(X, dtype=torch.float32)

        # Вычисление HOSVD
        core_tensor, factors = tensorly.decomposition.hosvd(X_tensor, ranks=rank, **kwargs)

        return core_tensor.numpy(), [factor.numpy() for factor in factors]