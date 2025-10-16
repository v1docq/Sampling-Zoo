import pandas as pd
import numpy as np
from sampling_strategies import SamplingStrategyFactory

# Создание данных
data = pd.DataFrame({
    'feature_1': np.random.randn(1000),
    'feature_2': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

# Использование фабрики для создания стратегии
factory = SamplingStrategyFactory()
strategy = factory.create_strategy('feature_clustering', n_clusters=3)

# Обучение и применение стратегии
strategy.fit(data[['feature_1', 'feature_2']], target=data['target'])
partitions = strategy.get_partitions()

print("Partitions created:")
for name, indices in partitions.items():
    print(f"{name}: {len(indices)} samples")