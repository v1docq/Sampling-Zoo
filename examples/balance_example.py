import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.api.api_main import SamplingStrategyFactory
from sklearn.datasets import make_classification



X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, n_clusters_per_class=1,
                           weights=[0.9, 0.1], flip_y=0, random_state=42)

data = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
target = pd.Series(y, name='target')

factory = SamplingStrategyFactory()
strategy = factory.create_strategy(strategy_type='balance', balance_method = 'smote', n_partitions=4)


# Обучаем сэмплер
strategy.fit(data, target)
partitions = strategy.get_partitions()

# 4. Анализ статистик в сбалансированных разделах
print("\nPartition statistics after balancing with SMOTE:")
for name, (feature_df, target_series) in partitions.items():
    print(f"\n{name} ({len(target_series)} samples):")
    