import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# from core.api.api_main import SamplingStrategyFactory
from core.sampling_strategies.delaunay_sempler import DelaunaySampler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


#  ГЕНЕРАЦИЯ ДАННЫХ
X_train, y_train = make_blobs(n_samples=600, centers=8, n_features=2, random_state=42, cluster_std=1.2)
X_train_df = pd.DataFrame(X_train, columns=['feat_1', 'feat_2'])
y_train_series = pd.Series(y_train)

# Равномерные данные для теста
rng = np.random.default_rng(42)
x_min, x_max = X_train[:, 0].min() - 3, X_train[:, 0].max() + 3
y_min, y_max = X_train[:, 1].min() - 3, X_train[:, 1].max() + 3

n_test_samples = 3000
X_test = rng.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(n_test_samples, 2))
X_test_df = pd.DataFrame(X_test, columns=['feat_1', 'feat_2'])


# ОБУЧЕНИЕ
# n_clusters=6 -> Строим триангуляцию на 6 точках
# n_partitions=10 -> Итоговых разделов хотим 10 (некоторые симплексы продублируются или разобьются)
sampler = DelaunaySampler(n_partitions=10, n_clusters=6, random_state=41, emptiness_threshold=0.01)
sampler.fit(X_train_df, y_train_series)
partitions_data = sampler.get_partitions(X_train_df, y_train_series)


# ПРЕДСКАЗАНИЕ 
test_simplex_indices = sampler.predict_partitions(X_test_df)


# ВИЗУАЛИЗАЦИЯ ОБУЧАЮЩИХ ДАННЫХ
fig, axes = plt.subplots(2, 2, figsize=(22, 18))
cmap = plt.cm.tab20

cluster_keys = sorted(partitions_data.keys(), key=lambda x: int(x.split('_')[1]))
for i, cluster_name in enumerate(cluster_keys):
    cluster_content = partitions_data[cluster_name]
    features = cluster_content['feature'].values
    # Парсим ID из имени
    cluster_id = int(cluster_name.split('_')[1])
    col = cmap(cluster_id % 20)
    axes[0, 0].scatter(features[:, 0], features[:, 1], c=[col], s=30, label=cluster_name, alpha=0.7, edgecolors='k', linewidth=0.3)

axes[0, 0].set_title('1. Train Data: Resulting Partitions')
axes[0, 0].legend(fontsize='x-small', loc='upper right')
axes[0, 0].grid(True, alpha=0.3)

# Рисуем точки
for i, cluster_name in enumerate(cluster_keys):
    features = partitions_data[cluster_name]['feature'].values
    cluster_id = int(cluster_name.split('_')[1])
    col = cmap(cluster_id % 20)
    axes[0, 1].scatter(features[:, 0], features[:, 1], c=[col], s=30, alpha=0.4)

# Рисуем триангуляцию
if sampler.centroids is not None and sampler.triang is not None:
    # Рисуем линии триангуляции
    axes[0, 1].triplot(sampler.centroids[:, 0], sampler.centroids[:, 1], sampler.triang.simplices, 
                       color='black', lw=1.5, alpha=0.8, label='Delaunay Edges')
    # Рисуем центроиды (вершины)
    axes[0, 1].plot(sampler.centroids[:, 0], sampler.centroids[:, 1], 'ko', markersize=8, label='Centroids (Vertices)')
    for key in sampler.kept_keys:
        if key < len(sampler.simplex_centers):
            center = sampler.simplex_centers[key]
            axes[0, 1].text(center[0], center[1], f"S{key}", fontsize=9, fontweight='bold', color='red',
                           ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
axes[0, 1].set_title('2. Train Data + Delaunay Triangulation')
axes[0, 1].set_xlim(x_min, x_max)
axes[0, 1].set_ylim(y_min, y_max)
axes[0, 1].legend(loc='upper right')




# ВИЗУАЛИЗАЦИЯ КАРТЫ ПРЕДСКАЗАНИЙ
axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c='gray', s=10, alpha=0.5, label='Random Noise')
axes[1, 0].set_title('3. Test Data: Random Uniform Noise')
axes[1, 0].set_xlim(x_min, x_max)
axes[1, 0].set_ylim(y_min, y_max)
axes[1, 0].grid(True, alpha=0.3)

unique_pred = np.unique(test_simplex_indices)

for label in unique_pred:
    mask = (test_simplex_indices == label)
    if np.sum(mask) == 0: continue
    
    subset = X_test[mask]
    col = cmap(label % 20)
    
    axes[1, 1].scatter(subset[:, 0], subset[:, 1], 
                       c=[col], s=15, alpha=0.6, label=f"Simplex {label}")

# Рисуем скелет триангуляции поверх
if sampler.centroids is not None:
    axes[1, 1].triplot(sampler.centroids[:, 0], sampler.centroids[:, 1], sampler.triang.simplices, 
                       color='black', lw=1, alpha=0.8)
    axes[1, 1].plot(sampler.centroids[:, 0], sampler.centroids[:, 1], 'kx', markersize=5)

axes[1, 1].set_title('4. Predict Result: Delaunay Simplexes + Nearest Center (Outside)')
axes[1, 1].set_xlim(x_min, x_max)
axes[1, 1].set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()