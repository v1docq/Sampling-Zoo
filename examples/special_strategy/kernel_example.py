import pandas as pd
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from core.sampling_strategies.kernel_sampler import KernelSampler
import torch
from itertools import product
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier

PATH_TO_TRAIN = "./dataset/census-income-train.csv"
PATH_TO_TEST = "./dataset/census-income-test.csv"
TARGET_COL = 'income'
KERNEL_SAMPLER_PARAMS = dict(energy_score=0,
                             density_score=0.3,
                             leverage_score=0.7,
                             anchors=1024)
TABNET_PARAMS = dict(max_epochs=100,
                             patience=10,
                     batch_size=1024,
                     virtual_batch_size=128)
ANCHORS_RANGE = [256, 512, 1024]

class BooleanKernelSampler:

    def __init__(self):
        self.sigma_range = [0.5, 1, 1.5]
        self.c_range = [(2, 3), (3, 4)]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sampler_model = KernelSampler(sample_size=20000,
                                           random_state=round(sigma * 2 + c[0] * 10))

    def get_data(self, type = torch.long):
        x_torch_train = torch.from_numpy(X_train_categorical_only.to_numpy()).to(device=self.device, dtype=type)
        x_torch_test = torch.from_numpy(X_test_categorical_only.to_numpy()).to(device=self.device, dtype=type)
        return x_torch_train, x_torch_test

    def pairwise_overlap(self, tensors):
        # для подсчета сколько общих индексов с разными настройками kernel-ов
        n = len(tensors)
        result = torch.zeros((n, n), dtype=torch.float32)

        tensors = [torch.unique(t) for t in tensors]

        for i in range(n):
            for j in range(n):
                a, b = tensors[i], tensors[j]

                inter = torch.isin(a, b).sum()
                union = len(a) + len(b) - inter

                result[i, j] = inter / union * 100

        return result

    def get_idx_inter(self):
        # смотрим пересечение засемплированных индексов для разных настроек kernel-ов с 1024 якорями

        x_torch_train = torch.from_numpy(X_train_cat_le.to_numpy()).to(device=self.device, dtype=torch.long)
        x_torch_train_num = torch.from_numpy(X_train_num.to_numpy()).to(device=self.device, dtype=torch.float32)
        x_torch_test = torch.from_numpy(X_test_cat_le.to_numpy()).to(device=self.device, dtype=torch.long)
        x_torch_test_num = torch.from_numpy(X_test_num.to_numpy()).to(device=self.device, dtype=torch.float32)

        idxs = []
        for sigma, c in product(self.sigma_range, self.c_range):
            kernels = [dict(name=f"mC_c{c}", kind="mC", c=mc, weight=1.0) for mc in c]
            kernels.extend(
                [dict(name="mdnf", kind="mdnf", sigma=sigma, weight=1.0), dict(name="match", kind="match", weight=0.5)])
            # это примерно соответствует balanced сэмплеру с преимущественно density score, потому что так лучше получается
            sampler = KernelSampler(sample_size=20000, kernels=kernels, energy_score=0, density_score=0.3,
                                    leverage_score=0.7,
                                    anchors=1024,
                                    random_state=round(sigma * 2 + c[0] * 10),
                                    w_f=torch.rand((x_torch_train.shape[1],), device=self.device, dtype=torch.float32))
            idx, A, embeddings_train = sampler.fit_sample(x_torch_train, y=None, sample_mode="topk")
            idxs.append(idx)

        return self.pairwise_overlap(idxs)


class KernelExperiment:
    def __init__(self, path_train, path_test):
        self.train_data = self.process_cat_data(self.load_data(path_train))
        self.test_data = self.process_cat_data(self.load_data(path_test))
        self.onh_data = self.get_onh_data(pd.concat([self.train_data, self.test_data]))
        self.model = self.get_model()

    def load_data(self, path: str):
        data = pd.read_csv(path)
        return data

    def split_data(self, data, target_col: str):
        X = data.drop(columns=[target_col])
        y = data[target_col]
        return X, y

    def get_onh_data(self, full_data):
        onh_data = pd.get_dummies(full_data, columns=self.categorical_cols, drop_first=False)
        # X_train_ohe = onh_data.iloc[:len(self.train_data)]
        # X_test_ohe = onh_data.iloc[len(self.train_data):]
        # X_full_le = se.copy()
        #
        # for col in self.categorical_cols:
        #     X_full_le[col] = (X_full[col].astype("category").cat.add_categories(["__MISSING__"]).
        #                       fillna("__MISSING__").cat.codes)
        # X_train_le = X_full_le.iloc[:len(X_train)]
        # X_test_le = X_full_le.iloc[len(X_train):]
        return onh_data

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
    def __init__(self,tabnet_params):
        self.model_params = tabnet_params

    def get_model(self):
        return TabNetClassifier(seed=42,verbose=0)

    def fit(self, X, y, cat_cols='auto'):
        self.model.fit(X,y,**self.model_params)
    def

        for anchors in [256, 512, 1024]:
            returned_idxs = []
            for sigma, c in product([0.5, 1, 1.5], [(2, 3), (3, 4)]):
                kernels = [dict(name=f"mC_c{c}", kind="mC", c=mc, weight=1.0) for mc in c]
                kernels.extend(
                    [dict(name="mdnf", kind="mdnf", sigma=sigma, weight=1.0),
                     dict(name="match", kind="match", weight=2)]
                )  # + [dict(name="rbf", kind="rbf", weight=1.0)] for numerical features processing
                # это примерно соответствует balanced сэмплеру с преимущественно density score, потому что так лучше получается
                sampler = KernelSampler(sample_size=20000, kernels=kernels, energy_score=0, density_score=0.7,
                                        leverage_score=0.3, anchors=anchors,
                                        random_state=int(sigma * 2 + c[0] * 5),
                                        w_f=torch.rand((x_torch_train.shape[1],), device=device, dtype=torch.float32))
                idx, A, embeddings_train = sampler.fit_sample(x_torch_train, x_torch_train_num, y=None)
                embeddings_test = sampler.get_embeddings(x_torch_test, X_num=x_torch_test_num, device=device)
                returned_idxs.append(idx)

                X_train_emb = embeddings_train.cpu().numpy()
                X_test_emb = embeddings_test.cpu().numpy()
                y_train_sample = y_train.iloc[idx.cpu().numpy()]

                scaler = StandardScaler()
                X_train_emb = scaler.fit_transform(X_train_emb)
                X_test_emb = scaler.transform(X_test_emb)

                # с PCA получается хуже
                # pca = PCA(n_components=0.92, svd_solver='full')
                # pca.fit(X_train_emb)
                # X_train_emb = pca.transform(X_train_emb)
                # X_test_emb = pca.transform(X_test_emb)

                print("Train shape:", X_train_emb.shape)
                print("Test shape:", X_test_emb.shape)

                # TabNet ожидает numpy float32
                X_train_emb = np.array(X_train_emb, dtype=np.float32)
                X_test_emb = np.array(X_test_emb, dtype=np.float32)
                y_train_sample = np.array(y_train_sample)
                y_test_np = np.array(y_test)

                # X_train_emb, X_val, y_train_sample, y_val = train_test_split(
                #     X_train_emb,
                #     y_train_sample,
                #     test_size=0.2,
                #     random_state=42,
                #     stratify=y_train_sample
                # )

                model_emb = TabNetClassifier(
                    seed=42,
                    verbose=0
                )



                pred_emb = model_emb.predict_proba(X_test_emb)[:, 1]
                auc_emb = roc_auc_score(y_test, pred_emb)
                pred_emb = model_emb.predict(X_test_emb)
                f1_emb = f1_score(y_test, pred_emb, average='weighted')
                f1_emb_macro = f1_score(y_test, pred_emb, average='macro')

                print(f"anchors: {anchors}, sigma: {sigma}, c: {c}")
                print("AUC with emb:", auc_emb)
                print("F1 with emb:", f1_emb)
                print("F1 macro with emb:", f1_emb_macro, '\n')

            idx_overlap = pairwise_overlap(returned_idxs)
            print(idx_overlap)

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
    pred_labels_cat = exp_handler.predict(X_test_categorical_only )
    pred_probs_cat = exp_handler.predict_proba(X_test_categorical_only )
    metrics_cat = exp_handler.get_metrics(pred_labels_cat, pred_probs_cat, y_test)


    # stage 3 - use boolean embeding for tabnet model
    boolean_handler = BooleanKernelSampler()
    x_torch_train_cat, x_torch_test_cat = boolean_handler.get_data(type=torch.long)
    x_torch_train_num, x_torch_test_num = boolean_handler.get_data(type=torch.float32)

    for anchors in ANCHORS_RANGE:
        returned_idxs = []
        for sigma, c in product(boolean_handler.sigma_range, boolean_handler.c_range):
            idx, A, embeddings_train = boolean_handler.sampler_model.fit_sample(x_torch_train, x_torch_train_num, y=None)
            embeddings_test = boolean_handler.sampler_model.get_embeddings(x_torch_test, X_num=x_torch_test_num, device=device)
            returned_idxs.append(idx)

            X_train_emb = embeddings_train.cpu().numpy()
            X_test_emb = embeddings_test.cpu().numpy()
            y_train_sample = y_train.iloc[idx.cpu().numpy()]

            scaler = StandardScaler()
            X_train_emb = scaler.fit_transform(X_train_emb)
            X_test_emb = scaler.transform(X_test_emb)
# # получаем label encoding представления, чтобы можно было проще сравнивать категории (в кернелах в итоге смотрится одинаковые ли числа, что ок)
# X_full = pd.concat([X_train, X_test])
# X_full_le = X_full.copy()
#
# for col in categorical_cols:
#     X_full_le[col] = (
#         X_full[col]
#         .astype("category")
#         .cat.add_categories(["__MISSING__"])
#         .fillna("__MISSING__")
#         .cat.codes
#     )
# X_train_le = X_full_le.iloc[:len(X_train)]
# X_test_le = X_full_le.iloc[len(X_train):]
# X_train_num = X_train_le[numeric_cols]
# X_test_num = X_test_le[numeric_cols]
#
# X_train_cat_le = X_train_le[categorical_cols]
# X_test_cat_le = X_test_le[categorical_cols]
#
# total_ohe_dim = 0
# for col in categorical_cols:
#     total_ohe_dim += int(X_full_le[col].max() + 1)
#
# # смотрим скалярные произведения между 10000 рандомными строками в OHE формате (то есть количество совпадений просто)
# import numpy as np
# import matplotlib.pyplot as plt
#
# X = X_train_categorical_only_ohe[:100000].to_numpy(dtype=np.float32)
# n_rows = X.shape[0]
# n_pairs = 10000
#
# rng = np.random.default_rng(42)
# pairs = rng.choice(n_rows, size=(n_pairs, 2), replace=True)
# row_i = X[pairs[:, 0]]
# row_j = X[pairs[:, 1]]
#
# dot_products = np.sum(row_i * row_j, axis=1)
#
# plt.hist(dot_products, bins=50)
# plt.title("Dot products Between Random Row Pairs")
# plt.xlabel("Dot product (shared active features)")
# plt.ylabel("Frequency")
# plt.show()
