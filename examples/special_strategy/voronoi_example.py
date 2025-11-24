import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
from typing import Dict, Any, Union

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.sampling_strategies.voronoi_sampler import VoronoiSampler

# Генерация данных
n_centers = 5
X_train, y_train = make_blobs(n_samples=500, centers=10, n_features=2, random_state=42, cluster_std=1.5)
X_train_df = pd.DataFrame(X_train, columns=['feat_1', 'feat_2'])
y_train_series = pd.Series(y_train)

# Равномерное распределение для теста
rng = np.random.default_rng(42)
x_min, x_max = X_train[:, 0].min() - 2, X_train[:, 0].max() + 2
y_min, y_max = X_train[:, 1].min() - 2, X_train[:, 1].max() + 2
n_test_samples = 2000
X_test = rng.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(n_test_samples, 2))
X_test_df = pd.DataFrame(X_test, columns=['feat_1', 'feat_2'])



# Обучение
sampler = VoronoiSampler(n_partitions=n_centers, random_state=42)
sampler.fit(X_train_df, y_train_series)
partitions_data = sampler.get_partitions(X_train_df, y_train_series)
# Предсказание 
test_labels = sampler.predict_partitions(X_test_df)



# ВИЗУАЛИЗАЦИЯ ОБУЧАЮЩИХ ДАННЫХ
vor = Voronoi(sampler.centroids)
fig, axes = plt.subplots(2, 2, figsize=(22, 18))

cluster_keys = sorted(partitions_data.keys())
for i, cluster_name in enumerate(cluster_keys):
    cluster_content = partitions_data[cluster_name]
    features = cluster_content['feature'].values
    cluster_id = int(cluster_name.split('_')[2])
    col = plt.cm.tab10(cluster_id % 10)
    axes[0, 0].scatter(features[:, 0], features[:, 1], c=[col], s=30, label=cluster_name, alpha=0.7, edgecolors='k', linewidth=0.5)

axes[0, 0].set_title('1. Train Data: Partitions')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Границы решетки
voronoi_plot_2d(vor, ax=axes[0, 1], show_vertices=False, line_colors='black', line_width=2, line_alpha=0.6, point_size=0)
for i, cluster_name in enumerate(cluster_keys):
    features = partitions_data[cluster_name]['feature'].values
    cluster_id = int(cluster_name.split('_')[2])
    col = plt.cm.tab10(cluster_id % 10)
    axes[0, 1].scatter(features[:, 0], features[:, 1], c=[col], s=30, alpha=0.6)
    # Центроиды
    centroid = sampler.centroids[cluster_id]
    axes[0, 1].text(centroid[0], centroid[1], str(cluster_id), fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

axes[0, 1].set_title('2. Train Data + Voronoi Cells')
axes[0, 1].set_xlim(x_min, x_max)
axes[0, 1].set_ylim(y_min, y_max)




# ВИЗУАЛИЗАЦИЯ КАРТЫ ПРЕДСКАЗАНИЙ
axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c='gray', s=10, alpha=0.5, label='Random Noise')
axes[1, 0].set_title('3. Test Data: Random Uniform Noise')
axes[1, 0].legend()
axes[1, 0].set_xlim(x_min, x_max)
axes[1, 0].set_ylim(y_min, y_max)
axes[1, 0].grid(True, alpha=0.3)

unique_pred = np.unique(test_labels)
for label in unique_pred:
    mask = (test_labels == label)
    subset = X_test[mask]
    col = plt.cm.tab10(label % 10)
    axes[1, 1].scatter(subset[:, 0], subset[:, 1], 
                       c=[col], s=15, alpha=0.6, label=f"Pred Cluster {label}")

# Границы решетки
voronoi_plot_2d(vor, ax=axes[1, 1], show_vertices=False, line_colors='black', line_width=2, line_alpha=0.8, point_size=0)

# центроиды
axes[1, 1].scatter(sampler.centroids[:, 0], sampler.centroids[:, 1], c='black', marker='X', s=100, label='Centroids')

axes[1, 1].set_title('4. Predict Result: Voronoi Tessellation Logic')
axes[1, 1].set_xlim(x_min, x_max)
axes[1, 1].set_ylim(y_min, y_max)
axes[1, 1].legend(fontsize='small', loc='upper right')

plt.tight_layout()
plt.show()