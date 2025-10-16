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
        self._init_constant()
        self.n_splits = n_splits
        self.method = method  # 'sequential', 'sliding_window', 'seasonal'
        self.window_size = kwargs.get('window_size', None)
        self.partitions = {}

    def _init_constant(self):
        self.series_id_column = 'series_id'
        self.time_column = 'timestamp'
        self.seasonal_period = 24
        self.method_map = {'sequential': self._sequential_split,
                           'sliding_window': self._sliding_window_split,
                           'seasonal': self._seasonal_split}

    def fit(self, data: pd.DataFrame, time_column: str = 'timestamp', series_id_column: str = 'series_id',
            **kwargs) -> 'TemporalSplitSampler':
        """
        Args:
            data: DataFrame с временными рядами
            time_column: Название колонки с временными метками
            series_id_column: Название колонки с идентификатором ряда
        """

        unique_series = data[series_id_column].unique()

        for series_id in unique_series:
            series_data = data[data[series_id_column] == series_id].sort_values(time_column)
            split_indices = self.method_map[self.method](series_data)
            self.partitions.update({series_id: split_indices})

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

    def _seasonal_split(self, data: pd.DataFrame) -> 'SeasonalSampler':
        # Извлекаем сезонную компоненту из временных меток
        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data['seasonal_component'] = data[self.time_column].dt.hour % self.seasonal_period
        seasonal_values = data['seasonal_component'].unique()
        season_val = {season:data[data['seasonal_component'] == season].index.values for season in seasonal_values}
        return season_val

    def get_partitions(self, data) -> Dict[Any, np.ndarray]:
        unique_series = data[self.series_id_column].unique()
        partition = {series_id: {k: data.iloc[self.partitions[series_id][k]] for k in self.partitions[series_id]}
                     for series_id in unique_series}
        return partition
