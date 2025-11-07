from enum import Enum
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class SamplingModels(Enum):
    balance_samplers = {
        # Методы передискретизации (Over-sampling)
        'smote': SMOTE,
        'adasyn': ADASYN,
        # Методы субдискретизации (Under-sampling)
        'rus': RandomUnderSampler,
        'enn': EditedNearestNeighbours,
        'tomek': TomekLinks,
        # Комбинированные методы
        'smoteenn': SMOTEENN,
        'smotetomek': SMOTETomek,
    }


class SupportingModels(Enum):
    difficulty_learner = {'classification': RandomForestClassifier,
                          'regression': RandomForestRegressor}
    scaling_models = {'scaler': StandardScaler}
    clustering_models = {'kmeans': KMeans, 'dbscan': DBSCAN, 'tsne': TSNE, 'pca': PCA}
