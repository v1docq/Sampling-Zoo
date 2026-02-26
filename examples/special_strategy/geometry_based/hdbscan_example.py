import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# from core.api.api_main import SamplingStrategyFactory
from core.sampling_strategies.hdbscan_sampler import HDBScanSampler

from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Generating visualization...")

# 1. Генерация данных
X, y_true = make_blobs(n_samples=500, centers=10, n_features=2, random_state=42, cluster_std=1.5)
X_df = pd.DataFrame(X, columns=['feat_1', 'feat_2'])
y_series = pd.Series(y_true)


# 2. Инициализация и обучение
sampler = HDBScanSampler(min_cluster_size=10, prob_threshold=0.3, one_cluster=True) # one_cluster=False для проверки soft threshold
sampler.fit(X_df, y_series)



# ВИЗУАЛИЗАЦИЯ КЛАСТЕРИЗАЦИИ ИСХОДНЫХ ДАННЫХ
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

labels = sampler.clusterer.labels_
unique_labels = set(labels)

for k in unique_labels:
    if k == -1:
        col = 'gray'
        label_text = 'Noise (Original)'
        alpha = 0.3
        marker = 'x'
    else:
        col = plt.cm.tab10(k % 10)
        label_text = f'Cluster {k}'
        alpha = 0.8
        marker = 'o'
    
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    axes[0].scatter(xy[:, 0], xy[:, 1], c=[col], s=40, label=label_text, alpha=alpha, marker=marker)

axes[0].set_title('1. Raw HDBSCAN Output (Contains Noise)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend(fontsize='small')
axes[0].grid(True, alpha=0.3)



# ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ ДЛЯ РАВНОМРНО РАСПРЕДЕЛЕНЫХ ТОЧЕК В ОБЛАСТИ
rng = np.random.default_rng(42)

# Определяем границы графика и сгенерируем расномерный датасет
x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
n_test_samples = 2000
X_test = rng.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(n_test_samples, 2))
X_test_df = pd.DataFrame(X_test, columns=['feat_1', 'feat_2'])

# Предсказание
test_labels = sampler.predict_partitions(X_test_df)

# Рисуем предсказанные точки
unique_pred = np.unique(test_labels)
for label in unique_pred:
    mask = (test_labels == label)
    subset = X_test[mask]
    col = plt.cm.tab10(label % 10)
    
    axes[1].scatter(subset[:, 0], subset[:, 1], 
                       c=[col], s=15, alpha=0.6, label=f"Pred Cluster {label}")

axes[1].set_title('2. Test Data: Random Uniform Noise')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend(fontsize='small')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()