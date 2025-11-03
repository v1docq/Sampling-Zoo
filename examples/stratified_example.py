import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.api.api_main import SamplingStrategyFactory

# Параметры распределений
n_samples = 10000
mu1, sigma1 = 5, 2  # параметры для первого признака
mu2, sigma2 = -3, 1.5  # параметры для второго признака

# Создание данных с заданными параметрами
data = pd.DataFrame({
    'feature_1': np.random.normal(mu1, sigma1, n_samples),
    'feature_2': np.random.normal(mu2, sigma2, n_samples), 
    'target': np.random.randint(0, 2, n_samples)
})

# Вывод исходных статистик
print("Original data statistics:")
print("Feature 1 (μ={}, σ={}):".format(mu1, sigma1))
print("  Mean: {:.3f}, Std: {:.3f}, Var: {:.3f}".format(
    data['feature_1'].mean(), 
    data['feature_1'].std(),
    data['feature_1'].var()
))
print("Feature 2 (μ={}, σ={}):".format(mu2, sigma2))
print("  Mean: {:.3f}, Std: {:.3f}, Var: {:.3f}\n".format(
    data['feature_2'].mean(), 
    data['feature_2'].std(),
    data['feature_2'].var()
))

# Использование фабрики для создания стратегии
factory = SamplingStrategyFactory()
strategy = factory.create_strategy('stratified_split', n_partitions=4)

# Обучение и применение стратегии
strategy.fit(data, strat_target=['feature_1', 'feature_2', 'target'])
partitions = strategy.get_partitions(data[['feature_1', 'feature_2']], target=data['target'])

print("Partition statistics:")
for name, part in partitions.items():
    indices = part['feature'].index.to_numpy()
    partition_data = data.iloc[indices]
    print(f"\n{name} ({len(indices)} samples):")
    print("Feature 1:")
    print("  Mean: {:.3f}, Std: {:.3f}, Var: {:.3f}".format(
        partition_data['feature_1'].mean(),
        partition_data['feature_1'].std(),
        partition_data['feature_1'].var()
    ))
    print("Feature 2:")
    print("  Mean: {:.3f}, Std: {:.3f}, Var: {:.3f}".format(
        partition_data['feature_2'].mean(),
        partition_data['feature_2'].std(),
        partition_data['feature_2'].var()
    ))

# for name, part in partitions.items():
#     idx = part['feature'].index.to_numpy()
#     y_true = np.asarray(part['target'])
#     y_pred = predictions[idx]
#     label = name

#     if idx.size == 0:
#         print(f"{label}: 0 samples (skipped)")
#         continue

#     errors = (y_true != y_pred).sum()
#     total = len(idx)
#     pct = errors / total
#     cm = confusion_matrix(y_true, y_pred)
#     print(f"{label}: {errors} errors out of {total} ({pct:.2%})")
#     print("Confusion matrix for", label)
#     print(cm)
