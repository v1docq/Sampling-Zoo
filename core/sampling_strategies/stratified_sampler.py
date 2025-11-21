from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, RepeatedMultilabelStratifiedKFold, \
    IterativeStratification
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils import check_random_state

from .base_sampler import BaseSampler, HierarchicalStratifiedMixin


class StratifiedSplitSampler(BaseSampler, HierarchicalStratifiedMixin):
    """
    Семплирование с сохранением распределений в указанных классах
    """

    def __init__(self, n_partitions: int = 5, random_state: int = 42, uniqueness_threshold: int = 0.3):
        BaseSampler.__init__(self, random_state=random_state)
        HierarchicalStratifiedMixin.__init__(
            self,
            n_splits=n_partitions,
            random_state=random_state,
            logger_name="StratifiedSplitSampler",
        )
        self.n_partitions = n_partitions
        self.uniqueness_threshold = uniqueness_threshold
        self.partitions = {}

    def fit(self, data: pd.DataFrame, target: list[str], data_target: list[str] = None):
        """
        Args:
            data: Матрица признаков или сырые данные
            strat_target: Переменные, для которых будет сохранено распределение
            data_target: Целевая переменная (опционально)
        """
        if len(data_target.shape) >= 2:
            mskf = MultilabelStratifiedKFold(n_splits=self.n_partitions, shuffle=True, random_state=self.random_state)
            # Кодируем целевой признак для классификации
            le = OneHotEncoder()
            data_target = le.fit_transform(data_target.values.reshape(-1, 1))
            data_target = data_target.toarray()
            target = [x for x in target if not x.__contains__('target')]
            processed_data = pd.DataFrame(index=data.index)
            splits = mskf.iterative_stratified_split(processed_data, data_target)
            for name in target:
                # Если число уникальных слагаемых в столбце больше порога, то делим его на квантили
                if len(data[name].unique()) / len(data) > self.uniqueness_threshold:
                    processed_data[name] = pd.qcut(data[name], q=self.n_partitions, labels=False, duplicates='drop')
                else:
                    processed_data[name] = data[name]

            multilabel = pd.get_dummies(processed_data)
            split_iteration = mskf.split(multilabel.values, data_target.values)
        else:
            stratified_sampler = AdvancedStratifiedSampler(n_splits=self.n_partitions, random_state=self.random_state)
            stratified_sampler.fit(data, data_target.values)
            split_iteration = stratified_sampler.get_partitions()

        for i, part_idx in enumerate(
                split_iteration.values() if isinstance(split_iteration, dict) else split_iteration):
            self.partitions[f'chunk_{i}'] = part_idx

        return self

    def get_partitions(self, data, target) -> Dict[Any, np.ndarray]:
        partition = {cluster: dict(feature=data.iloc[idx],
                                   target=target.iloc[idx]) for cluster, idx in self.partitions.items()}
        return partition

    def check_partitions(self, partitions, data):
        print("Partition statistics:")
        feature_names = data.columns.to_list()
        feature_names = [x for x in feature_names if not x.__contains__('target')]
        for name, part in partitions.items():
            for feat in feature_names:
                indices = part['feature'].index.to_numpy()
                partition_data = data.iloc[indices]
                partition_data = partition_data[feat]
                print(f"\n{name} ({len(indices)} samples):")
                print(f"{feat}:")
                print("  Mean: {:.3f}, Std: {:.3f}, Var: {:.3f}".format(
                    partition_data.mean(),
                    partition_data.std(),
                    partition_data.var()))


class AdvancedStratifiedSampler(BaseSampler, HierarchicalStratifiedMixin):
    """
    Продвинутый стратифицированный семплер для многоклассовых задач
    с гарантированным присутствием всех классов в каждом фолде
    """

    def __init__(self, n_partitions: int = 100, random_state: int = 42):
        BaseSampler.__init__(self, random_state=random_state)
        HierarchicalStratifiedMixin.__init__(self, n_partitions=n_partitions, random_state=random_state)
        self.partitions = {}

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], min_samples_per_class: int = 2):
        folds = self.hierarchical_stratified_split(data, np.asarray(target), min_samples_per_class)
        self.partitions = {f'chunk_{i}': fold_indices for i, fold_indices in enumerate(folds)}
        return self

    def get_partitions(self) -> Dict[str, np.ndarray]:
        return self.partitions

    def get_fold_statistics(self, folds: List[np.ndarray], y: np.ndarray) -> pd.DataFrame:
        stats = []

        for i, fold_indices in enumerate(folds):
            fold_classes = Counter(y[fold_indices])
            stats.append({
                'fold': i,
                'test_size': len(fold_indices),
                'test_n_classes': len(fold_classes),
                'test_min_class': min(fold_classes.values()) if fold_classes else 0,
                'test_max_class': max(fold_classes.values()) if fold_classes else 0
            })

        return pd.DataFrame(stats)


class RegressionStratifiedSampler(BaseSampler, HierarchicalStratifiedMixin):
    """Стратифицированный семплер для регрессионных целей.

    Непрерывные цели дискретизируются на ``n_bins`` перед стратификацией.Поддерживает стратегии
    ``uniform``/``quantile``/``kmeans`` или квантование через `pandas.qcut` для квантильной стратегии.
    """

    def __init__(
            self,
            n_bins: int = 5,
            encode: str = "ordinal",
            strategy: str = "quantile",
            n_partitions: int = 5,
            random_state: int = 42,
            use_advanced: bool = True,
    ) -> None:
        BaseSampler.__init__(self, random_state=random_state)
        HierarchicalStratifiedMixin.__init__(
            self,
            n_partitions=n_partitions,
            random_state=random_state,
            logger_name="RegressionStratifiedSampler",
        )
        self.binning_model = RegressionStratifiedBinning(n_partitions=n_partitions, random_state=random_state)
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.use_advanced = use_advanced
        self.partitions_: Dict[str, np.ndarray] = {}
        self.bin_edges_: Optional[np.ndarray] = None
        self.binned_target_: Optional[np.ndarray] = None
        self.discretizer_: Optional[KBinsDiscretizer] = None

    def _discretize_target(self, target: Union[pd.Series, np.ndarray]) -> np.ndarray:
        target_series = pd.Series(target)

        if self.strategy == "quantile":
            binned, bin_edges = pd.qcut(
                target_series,
                q=self.n_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            self.bin_edges_ = bin_edges
            return binned.to_numpy(dtype=int)

        if self.strategy not in {"uniform", "kmeans"}:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. Use one of ['uniform', 'quantile', 'kmeans']."
            )

        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode=self.encode,
            strategy=self.strategy,
            random_state=self.random_state,
        )
        transformed = discretizer.fit_transform(target_series.to_numpy().reshape(-1, 1))

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        # Приводим к одномерным меткам даже при onehot-кодировании
        binned_target = np.argmax(transformed, axis=1) if transformed.ndim > 1 else transformed.ravel()
        self.bin_edges_ = discretizer.bin_edges_[0]
        self.discretizer_ = discretizer
        return binned_target.astype(int)

    def fit(self, data: pd.DataFrame, data_target: Union[pd.Series, np.ndarray], target=None):
        discrete_target = self._discretize_target(data_target)
        self.binned_target_ = discrete_target

        if self.use_advanced:
            stratified_sampler = AdvancedStratifiedSampler(n_partitions=self.n_partitions,
                                                           random_state=self.random_state)
            placeholder = pd.DataFrame(index=data.index)
            stratified_sampler.fit(placeholder, discrete_target, min_samples_per_class=1)
            self.partitions = stratified_sampler.get_partitions()
        else:
            partitions = {}
            for i, (_, test_idx) in enumerate(
                    self.stratification_model.split(np.zeros_like(discrete_target), discrete_target)
            ):
                partitions[f"chunk_{i}"] = test_idx
            self.partitions = partitions

        return self

    def get_partitions(self, data, target) -> Dict[str, np.ndarray]:
        partition = {cluster: dict(feature=data.iloc[idx],
                                   target=target.iloc[idx]) for cluster, idx in self.partitions.items()}
        return partition



class RegressionStratifiedBinning(BaseSampler, HierarchicalStratifiedMixin):
    SUPPORTED_BIN_RULES = ("freedman_diaconis", "scott", "quantile")

    def __init__(self, n_partitions: int = 5, random_state: int = 42, bin_rule: str = "freedman_diaconis"):
        if bin_rule not in self.SUPPORTED_BIN_RULES:
            raise ValueError(f"bin_rule должен быть одним из {self.SUPPORTED_BIN_RULES}, получено: {bin_rule}")

        BaseSampler.__init__(self, random_state=random_state)
        HierarchicalStratifiedMixin.__init__(self, n_partitions=n_partitions, random_state=random_state,
                                             logger_name="RegressionStratifiedSampler")
        self.bin_rule = bin_rule
        self.partitions: Dict[str, np.ndarray] = {}
        self.bin_edges_: Optional[np.ndarray] = None
        self.binned_target_: Optional[np.ndarray] = None

    def fit(self, data: pd.DataFrame, data_target: Union[pd.Series, np.ndarray], min_samples_per_class: int = 1,
            target=None):
        y = np.asarray(data_target, dtype=float)
        if y.ndim != 1:
            y = y.reshape(-1)

        binned_target, bin_edges = self._bin_target(y, min_samples_per_class)
        self.bin_edges_ = bin_edges
        self.binned_target_ = binned_target

        self.logger.info(f"Правило биннинга: {self.bin_rule}")
        self.logger.info(f"Число бинов после объединения: {len(np.unique(binned_target))}")

        folds = self.hierarchical_stratified_split(data, binned_target, min_samples_per_class)
        self.partitions = {f'chunk_{i}': fold_indices for i, fold_indices in enumerate(folds)}
        return self

    def _bin_target(self, y: np.ndarray, min_samples_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
        bin_edges = self._calculate_bin_edges(y)
        initial_labels = np.digitize(y, bins=bin_edges[1:-1], right=False)
        merged_labels, merged_edges = self._merge_small_bins(initial_labels, bin_edges, min_samples_per_class)
        return merged_labels, merged_edges

    def _calculate_bin_edges(self, y: np.ndarray) -> np.ndarray:
        y_min, y_max = np.min(y), np.max(y)
        if y_min == y_max:
            return np.array([y_min, y_max + 1e-9])

        if self.bin_rule == "quantile":
            n_bins = max(2, min(len(np.unique(y)), self.n_splits))
            edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([y_min, y_max])
            return edges

        width = self._calculate_bin_width(y)
        if not np.isfinite(width) or width <= 0:
            return np.array([y_min, y_max])

        n_bins = max(1, int(np.ceil((y_max - y_min) / width)))
        return np.linspace(y_min, y_max, n_bins + 1)

    def _calculate_bin_width(self, y: np.ndarray) -> float:
        n = len(y)
        if self.bin_rule == "freedman_diaconis":
            q75, q25 = np.percentile(y, [75, 25])
            iqr = q75 - q25
            return 2 * iqr / np.cbrt(n) if iqr > 0 else np.inf
        elif self.bin_rule == "scott":
            std = np.std(y, ddof=1)
            return 3.5 * std / np.cbrt(n) if std > 0 else np.inf
        else:
            return np.inf

    def _merge_small_bins(self, labels: np.ndarray, edges: np.ndarray, min_samples: int) -> Tuple[
        np.ndarray, np.ndarray]:
        bins = []
        for i in range(len(edges) - 1):
            idx = np.where(labels == i)[0]
            bins.append({"indices": idx, "start": edges[i], "end": edges[i + 1]})

        i = 0
        while i < len(bins):
            if len(bins[i]["indices"]) >= min_samples or len(bins) == 1:
                i += 1
                continue

            if i == 0:
                neighbor = 1
            elif i == len(bins) - 1:
                neighbor = i - 1
            else:
                left_size = len(bins[i - 1]["indices"])
                right_size = len(bins[i + 1]["indices"])
                neighbor = i - 1 if left_size <= right_size else i + 1

            bins[neighbor]["indices"] = np.concatenate((bins[neighbor]["indices"], bins[i]["indices"]))
            bins[neighbor]["start"] = min(bins[neighbor]["start"], bins[i]["start"])
            bins[neighbor]["end"] = max(bins[neighbor]["end"], bins[i]["end"])
            bins.pop(i)

            if neighbor > i:
                neighbor -= 1
            i = max(neighbor, 0)

        merged_labels = np.zeros_like(labels)
        merged_edges = [bins[0]["start"]]
        for idx, bin_info in enumerate(bins):
            merged_labels[bin_info["indices"]] = idx
            merged_edges.append(bin_info["end"])

        return merged_labels, np.array(merged_edges)

    def get_partitions(self, data, target) -> Dict[str, np.ndarray]:
        partition = {cluster: dict(feature=data.iloc[idx],
                                   target=target.iloc[idx]) for cluster, idx in self.partitions.items()}
        return partition
