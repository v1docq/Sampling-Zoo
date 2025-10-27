import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.api.api_main import SamplingStrategyFactory

# Создание данных
data = pd.DataFrame({
    'feature_1': np.random.normal(10000),
    'feature_2': np.random.randn(10000),
    'target': np.random.randint(1000)
})

# Использование фабрики для создания стратегии
factory = SamplingStrategyFactory()
strategy = factory.create_strategy('stratified_split', n_partitions=4)
# print(data[['feature_1', 'feature_2']])
# Обучение и применение стратегии
strategy.fit(data, strat_target=['feature_1', 'feature_2', 'target'])
partitions = strategy.get_partitions(data[['feature_1', 'feature_2']], target=data['target'])

print("Partitions created:")
for name, indices in partitions.items():
    print(f"{name}: {len(indices)} samples")

