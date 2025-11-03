from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd


class BaseSampler(ABC):
    """
    Абстрактный базовый класс для всех стратегий семплирования
    """

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.partitions_ = None
        # # Игнорируем дополнительные аргументы для обратной совместимости
        # self.__dict__.update(locals())

    @abstractmethod
    def fit(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'BaseSampler':
        """
        Обучение семплера на данных

        Args:
            data: Входные данные
            **kwargs: Дополнительные параметры

        Returns:
            self: Обученный семплер
        """
        pass

    @abstractmethod
    def get_partitions(self) -> Dict[Any, np.ndarray]:
        """
        Возвращает индексы разделов

        Returns:
            Dict с индексами для каждого раздела
        """
        pass

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[Any, Union[np.ndarray, pd.DataFrame]]:
        """
        Преобразует данные в разделы

        Args:
            data: Исходные данные

        Returns:
            Dict с разделенными данными
        """
        partitions_indices = self.get_partitions()
        result = {}

        for partition_name, indices in partitions_indices.items():
            if isinstance(data, pd.DataFrame):
                result[partition_name] = data.iloc[indices]
            else:
                result[partition_name] = data[indices]

        return result

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> Dict[
        Any, Union[np.ndarray, pd.DataFrame]]:
        """
        Обучение и преобразование за один шаг
        """
        self.fit(data, **kwargs)
        return self.transform(data)