from enum import Enum
from sklearn.datasets import make_classification, make_regression


class SyntDataset(Enum):
    DEFAULT_CLF_DATASET_PARAMS = dict(n_samples=1000, n_features=2, n_informative=2,
                                      n_redundant=0, n_classes=2, n_clusters_per_class=1,
                                      weights=[0.9, 0.1], flip_y=0, random_state=42)
    DEFAULT_REG_DATASET_PARAMS = dict(n_samples=1000, n_features=2, n_informative=2)
    DATASET_GENERATORS = dict(classification=make_classification,
                              regression=make_regression)
    DATASET_DEFAULT_PARAMS = dict(classification=DEFAULT_CLF_DATASET_PARAMS,
                                  regression=DEFAULT_REG_DATASET_PARAMS)


class AmlbExperimentDataset(Enum):
    CLF_DATASET = [
        {
            'name': 'electricity',
            'openml_id': 151,
            'samples': 45312,
            'features': 8,
            'type': 'classification'
        },
        {
            'name': 'covertype',
            'openml_id': 1596,
            'samples': 581012,
            'features': 54,
            'type': 'classification'
        },
        {
            'name': 'adult',
            'openml_id': 1590,
            'samples': 48842,
            'features': 14,
            'type': 'classification'
        },
        {
            'name': 'numerai28.6',
            'openml_id': 23517,
            'samples': 96320,
            'features': 21,
            'type': 'classification'
        },
        {
            'name': 'bank-marketing',
            'openml_id': 1461,
            'samples': 45211,
            'features': 16,
            'type': 'classification'
        }
    ]
    REG_DATASET = [
        {
            'name': 'black friday',
            'openml_id': 531,
            'samples': 506,
            'features': 13,
            'type': 'regression'
        },
        {
            'name': 'diabetes',
            'openml_id': 847,
            'samples': 442,
            'features': 10,
            'type': 'regression'
        },
        {
            'name': 'california',
            'openml_id': 43905,
            'samples': 20640,
            'features': 8,
            'type': 'regression'
        },
        {
            'name': 'house_prices',
            'openml_id': 42705,
            'samples': 1460,
            'features': 80,
            'type': 'regression'
        },
        {
            'name': 'medical_charges',
            'openml_id': 42728,
            'samples': 163065,
            'features': 5,
            'type': 'regression'
        }
    ]
    AMLB_CUSTOM_DATASET = [{'name': 'covtype-normalized',
                            'type': 'classification',
                            'target': 'class',
                            'path': './dataset/covtype-normalized.csv'},
                           {'name': 'kddcup',
                            'type': 'classification',
                            'target': 'label',
                            'path': './dataset/kddcup.csv'},
                           {'name': 'airlines',
                            'type': 'regression',
                            'target': 'DepDelay',
                            'path': './dataset/airlines_train_regression_10M.csv'},
                           {'name': 'sf-police-incidents',
                            'type': 'classification',
                            'target': 'ViolentCrime',
                            'path': './dataset/sf-police-incidents.csv'}]
    AMLB_EXPERIMENT_RESULTS = {
        'adult': {'accuracy': 0.85, 'f1_macro': 0.65},
        'covertype': {'accuracy': 0.70, 'f1_macro': 0.55},
        'electricity': {'accuracy': 0.75, 'f1_macro': 0.60},
        'boston': {'rmse': 4.5, 'r2': 0.75},
        'california': {'rmse': 0.8, 'r2': 0.65}
    }
    FEDOT_BASELINE_PRESET = dict(timeout=10, preset='best_quality', cv_folds=3)
    FEDOT_PRESET = {'timeout': 1,  # Меньше timeout для каждой модели
                    'preset': 'best_quality',
                    'cv_folds': 2,
                    'logging_level': 20,
                    'with_tuning': False,
                    'pop_size': 5,
                    'num_of_generations': 10,
                    'n_jobs': 1,
                    'metric': 'f1'
                    }
    FEDOT_MODELS_FOR_CLF = [
        # 'bernb',
        'catboost',
        # 'dt',
        'fast_ica',
        'isolation_forest_class',
        'knn',
        'lgbm',
        'logit',
        'mlp',
        'normalization',
        'pca',
        # 'poly_features',
        # 'qda',
        'resample',
        'rf',
        'scaling',
        'xgboost'
    ]
    FEDOT_MODELS_FOR_REG = [
        'ridge',
        'rfr',
        'lgbmreg',
        'treg',
        'knnreg',
        'scaling'
    ]
    SAMPLING_PRESET = {'strategy': 'stratified',
                       "n_partitions": 3,
                       }
    REGRESSION_SAMPLING_PRESET = {
        'strategy': 'regression_stratified',
        'n_partitions': 3,
        'n_bins': 5,
        'encode': 'ordinal',
        'binning_strategy': 'quantile'
    }
