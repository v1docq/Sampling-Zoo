import numpy as np
from sklearn.utils.extmath import randomized_svd

class RandomizedSVDBackend:
    def __init__(self, oversample_factor=0, power_iterations=0, random_state=None):
        self.oversample_factor = oversample_factor 
        self.power_iterations = power_iterations   
        self.random_state = random_state

    def compute_basis(self, X, rank, **kwargs):
        """
        Вычисляет приближённый SVD с oversampling и power iterations. В данный момент реализован только randomized SVD из sklearn.
        Arguments:
        X: входная матрица  
        rank: целевой ранг (int или float в (0,1], обозначающий долю от min(n_samples, n_features))
        **kwargs: дополнительные параметры для randomized_svd
        Returns:
        Q: approximate range basis (ортогональная матрица, аппроксимирующая X)
        S: singular values
        Vt: right singular vectors
        """
        if isinstance(rank, float):
            rank = int(rank * min(X.shape))

        n_components = min(rank + self.oversample_factor, X.shape[1])  # Oversampling
        
        # Randomized SVD с power iterations
        U, S, Vt = randomized_svd(
            X,
            n_components=n_components,
            n_iter=self.power_iterations,
            random_state=self.random_state,
            **kwargs
        )
        
        Q = U[:, :rank] 
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        return Q, S, Vt