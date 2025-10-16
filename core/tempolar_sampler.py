import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from .base_sampler import BaseSampler


class TemporalSplitSampler(BaseSampler):
    """
    Стратегия временного разбиения для временных рядов
    """

    def __init__(self, n_splits: int = 4, method: str = 'sequential', **kwargs):
        super().__init__(**kwargs)
        self.n_splits = n_splits
        self.method = method  # 'sequential', 'sliding_window'
        self.window_size = kwargs.get('window_size', None)

    def fit(self, data: pd.DataFrame, time_column: str = 'timestamp',
            series_id_column: str = 'series_id', **kwargs) -> 'TemporalSplitSampler':
        """
        Args:
            data: DataFrame с временными рядами
            time_column: Название колонки с временными метками
            series_id_column: Название колонки с идентификатором ряда
        """
        self.partitions_ = {}

        unique_series = data[series_id_column].unique()

        for series_id in unique_series:
            series_data = data[data[series_id_column] == series_id].sort_values(time_column)

            if self.method == 'sequential':
                split_indices = self._sequential_split(series_data)
            elif self.method == 'sliding_window':
                split_indices = self._sliding_window_split(series_data)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Агрегируем индексы по разделам
            for split_id, indices in split_indices.items():
                if split_id not in self.partitions_:
                    self.partitions_[split_id] = []
                self.partitions_[split_id].extend(indices)

        # Преобразуем списки в массивы
        for split_id in self.partitions_:
            self.partitions_[split_id] = np.array(self.partitions_[split_id])

        return self

    def _sequential_split(self, series_data: pd.DataFrame) -> Dict[int, list]:
        """Последовательное разбиение на равные отрезки"""
        n_points = len(series_data)
        split_points = np.linspace(0, n_points, self.n_splits + 1, dtype=int)

        splits = {}
        for i in range(self.n_splits):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            splits[i] = series_data.iloc[start_idx:end_idx].index.tolist()

        return splits

    def _sliding_window_split(self, series_data: pd.DataFrame) -> Dict[int, list]:
        """Разбиение скользящим окном"""
        if not self.window_size:
            self.window_size = len(series_data) // self.n_splits

        splits = {}
        for i in range(self.n_splits):
            start_idx = i * self.window_size
            end_idx = start_idx + self.window_size

            if end_idx <= len(series_data):
                splits[i] = series_data.iloc[start_idx:end_idx].index.tolist()

        return splits

    def get_partitions(self) -> Dict[Any, np.ndarray]:
        return self.partitions_


class SeasonalSampler(BaseSampler):
    """
    Семплирование на основе сезонных паттернов
    """

    def __init__(self, seasonal_period: int = 24, **kwargs):
        super().__init__(**kwargs)
        self.seasonal_period = seasonal_period

    def fit(self, data: pd.DataFrame, time_column: str = 'timestamp', **kwargs) -> 'SeasonalSampler':
        # Извлекаем сезонную компоненту из временных меток
        data[time_column] = pd.to_datetime(data[time_column])
        data['seasonal_component'] = data[time_column].dt.hour % self.seasonal_period

        self.partitions_ = {}
        seasonal_values = data['seasonal_component'].unique()

        for season in seasonal_values:
            self.partitions_[season] = data[data['seasonal_component'] == season].index.values

        return self

    def get_partitions(self) -> Dict[Any, np.ndarray]:
        return self.partitions_
