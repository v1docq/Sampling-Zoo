import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb
from core.sampling_strategies.kernel_sampler import KernelSampler
import torch
from itertools import product
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier

PATH_TO_TRAIN = "./dataset/census-income-train.csv"
PATH_TO_TEST = "./dataset/census-income-test.csv"
TARGET_COL = 'income'
KERNEL_SAMPLER_PARAMS = dict(energy_score=0,
                             density_score=0.3,
                             leverage_score=0.7)
TABNET_PARAMS = dict(max_epochs=100,
                     patience=10,
                     batch_size=1024,
                     virtual_batch_size=128)
ANCHORS_RANGE = [
    0.01,
    # 0.05,
    # 0.1,
]
SIGMA_RANGE = [
    0.5,
    # 1,
    # 1.5
]
C_RANGE = [
    (2, 3),
    # (3, 4)
]


class BooleanKernelSampler:

    def __init__(self):
        self.sigma_range = SIGMA_RANGE
        self.c_range = C_RANGE
        self.anchors_range = ANCHORS_RANGE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sampler_params = KERNEL_SAMPLER_PARAMS

    def get_data(self, X_train, X_test, type=torch.long):
        x_torch_train = torch.from_numpy(X_train.to_numpy()).to(device=self.device, dtype=type)
        x_torch_test = torch.from_numpy(X_test.to_numpy()).to(device=self.device, dtype=type)
        return x_torch_train, x_torch_test

    def get_embeddings(self, x_torch_train, x_torch_test, scaling: bool = True):
        data_dict = dict()
        for anchors in self.anchors_range:
            returned_idxs = []
            anchor_size = round(anchors*x_torch_train.shape[0])
            for sigma, c in product(self.sigma_range, self.c_range):
                kernels = [dict(name=f"mC_c{c}", kind="mC", c=mc, weight=1.0) for mc in c]
                kernels.extend(
                    [dict(name="mdnf", kind="mdnf", sigma=sigma, weight=1.0),
                     dict(name="match", kind="match", weight=2)])
                w_f = torch.rand((x_torch_train.shape[1],), device=self.device, dtype=torch.float32)
                sampler_model = KernelSampler(sample_size=20000,
                                              kernels=kernels,
                                              anchors=anchor_size,
                                              random_state=int(sigma * 2 + c[0] * 5),
                                              w_f=w_f, **self.sampler_params)
                idx, A, embeddings_train = sampler_model.fit_sample(x_torch_train,
                                                                    x_torch_train_num,
                                                                    y=None)
                embeddings_test = sampler_model.get_embeddings(x_torch_test, X_num=x_torch_test_num, device=self.device)
                returned_idxs.append(idx)

                if scaling:
                    scaler = StandardScaler()
                    X_train_emb = scaler.fit_transform(embeddings_train.cpu().numpy())
                    X_test_emb = scaler.transform(embeddings_test.cpu().numpy())
                else:
                    X_train_emb = embeddings_train.cpu().numpy()
                    X_test_emb = embeddings_test.cpu().numpy()
                y_train_sample = y_train.iloc[idx.cpu().numpy()]
                # TabNet ожидает numpy float32
                data_dict.update({f'num_anchors - {anchors}, mC_c - {c}, sigma - {sigma}':
                                      dict(x_train=np.array(X_train_emb, dtype=np.float32),
                                           y_train=np.array(y_train_sample, dtype=np.float32),
                                           x_test=np.array(X_test_emb, dtype=np.float32),
                                           y_test=np.array(y_test))})

        return data_dict


class KernelExperiment:
    def __init__(self, path_train, path_test):
        self.train_data = self.process_cat_data(self.load_data(path_train))
        self.test_data = self.process_cat_data(self.load_data(path_test))
        self.full_data = pd.concat([self.train_data, self.test_data])
        self.get_encoded_data()
        self.get_onh_data()
        self.model = self.get_model()

    def load_data(self, path: str):
        data = pd.read_csv(path)
        return data

    def split_data(self, data, target_col: str):
        X = data.drop(columns=[target_col])
        y = data[target_col]
        return X, y

    def get_onh_data(self):
        self.onh_data = pd.get_dummies(self.full_data, columns=self.categorical_cols, drop_first=False)

    def get_encoded_data(self):
        self.encoded_data = self.full_data.copy()
        for col in self.categorical_cols:
            self.encoded_data[col] = (self.encoded_data[col].astype("category").cat.add_categories(["__MISSING__"]).
                                      fillna("__MISSING__").cat.codes)

    def process_cat_data(self, data):
        self.categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numeric_cols = data.select_dtypes(exclude=["object", "category"]).columns.tolist()
        for col in self.categorical_cols:
            data[col] = data[col].astype("category")
            data[col] = data[col].astype("category")
        return data

    def get_model(self):
        return lgb.LGBMClassifier(random_state=42, verbosity=-1)

    def fit(self, X, y, cat_cols='auto'):
        self.model.fit(X, y, categorical_feature=cat_cols)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

    def get_metrics(self, pred, pred_proba, target):
        metric_dict = {'roc_auc': roc_auc_score(target, pred_proba),
                       'f1_weighted': f1_score(target, pred, average='weighted'),
                       'f1_macro': f1_score(target, pred, average='weighted')}
        print("AUC full:", metric_dict['roc_auc'])
        print("F1 macro:", metric_dict['f1_macro'])
        print("F1 weighted:", metric_dict['f1_weighted'])
        return metric_dict


class TabnetExperiment(KernelExperiment):
    def __init__(self, path_train, path_test, tabnet_params):
        super().__init__(path_train, path_test)
        self.model_params = tabnet_params

    def get_model(self):
        return TabNetClassifier(seed=42, verbose=1)

    def fit(self, X, y, cat_cols='auto'):
        self.model.fit(X, y, **self.model_params)


if __name__ == "__main__":
    exp_handler = KernelExperiment(path_test=PATH_TO_TEST, path_train=PATH_TO_TRAIN)
    X_train, y_train = exp_handler.split_data(exp_handler.train_data, TARGET_COL)
    X_test, y_test = exp_handler.split_data(exp_handler.test_data, TARGET_COL)
    fitted_model_baseline = exp_handler.fit(X_train, y_train, exp_handler.categorical_cols)
    pred_labels = exp_handler.predict(X_test)
    pred_probs = exp_handler.predict_proba(X_test)
    metrics = exp_handler.get_metrics(pred_labels, pred_probs, y_test)

    # second stage - split data on two part, with only numerical columns and with only cat columns
    X_train_ohe = exp_handler.onh_data.iloc[:len(exp_handler.train_data)]
    X_test_ohe = exp_handler.onh_data.iloc[len(exp_handler.train_data):]
    X_train_num = exp_handler.full_data.iloc[:len(exp_handler.train_data)][exp_handler.numeric_cols]
    X_test_num = exp_handler.onh_data.iloc[len(exp_handler.train_data):][exp_handler.numeric_cols]

    binary_cols = [col for col in X_train_ohe.columns if col not in exp_handler.numeric_cols]
    category_cols = [col for col in X_train.columns if col not in exp_handler.numeric_cols]

    # use only binary columns
    X_train_categorical_only_ohe = X_train_ohe[binary_cols]
    X_test_categorical_only_ohe = X_test_ohe[binary_cols]
    fitted_model_onh = exp_handler.fit(X_train_categorical_only_ohe, y_train)
    pred_labels_onh = exp_handler.predict(X_test_categorical_only_ohe)
    pred_probs_onh = exp_handler.predict_proba(X_test_categorical_only_ohe)
    metrics_onh = exp_handler.get_metrics(pred_labels_onh, pred_probs_onh, y_test)

    # use only category colums
    X_train_categorical_only = X_train[category_cols]
    X_test_categorical_only = X_test[category_cols]
    fitted_model_cat = exp_handler.fit(X_train_categorical_only, y_train)
    pred_labels_cat = exp_handler.predict(X_test_categorical_only)
    pred_probs_cat = exp_handler.predict_proba(X_test_categorical_only)
    metrics_cat = exp_handler.get_metrics(pred_labels_cat, pred_probs_cat, y_test)

    # stage 3 - use boolean embeding for tabnet model
    X_train_encoded = exp_handler.encoded_data.iloc[:len(X_train)]
    X_test_encoded = exp_handler.encoded_data.iloc[len(X_train):]
    X_train_encoded_categorical_only = X_train_encoded[exp_handler.categorical_cols]
    X_test_encoded_categorical_only = X_test_encoded[exp_handler.categorical_cols]
    boolean_handler = BooleanKernelSampler()
    x_torch_train_cat, x_torch_test_cat = boolean_handler.get_data(X_train_encoded_categorical_only,
                                                                   X_test_encoded_categorical_only,
                                                                   type=torch.long)
    x_torch_train_num, x_torch_test_num = boolean_handler.get_data(X_train_num,
                                                                   X_test_num,
                                                                   type=torch.float32)
    generated_kernels = boolean_handler.get_embeddings(x_torch_train_cat, x_torch_test_cat, scaling=True)
    for boolean_kernel_params, embeddings in generated_kernels.items():
        tabnet_handler = TabnetExperiment(path_test=PATH_TO_TEST, path_train=PATH_TO_TRAIN, tabnet_params=TABNET_PARAMS)
        fitted_tabnet = tabnet_handler.fit(embeddings['x_train'], embeddings['y_train'], exp_handler.categorical_cols)
        pred_labels = tabnet_handler.predict(embeddings['x_test'])
        pred_probs = tabnet_handler.predict_proba(embeddings['x_test'])
        metrics_tabnet = tabnet_handler.get_metrics(pred_labels, pred_probs, embeddings['y_test'])
        _ = 1
