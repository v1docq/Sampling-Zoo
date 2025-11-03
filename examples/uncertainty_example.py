import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.api.api_main import SamplingStrategyFactory

# Создание данных: линейная комбинация признаков задаёт вероятность класса
np.random.seed(42)
n = 1000
feature_1 = np.random.randn(n)
feature_2 = np.random.randn(n) * 0.5
logit = 2.0 * feature_1 - 1.0 * feature_2 + 0.2 * np.random.randn(n)
prob = 1 / (1 + np.exp(-logit))
target = (prob > 0.5).astype(int)

data = pd.DataFrame({
    'feature_1': feature_1,
    'feature_2': feature_2,
    'target': target
})

features = data[['feature_1', 'feature_2']]

# Создаём стратегию UncertaintySampler
factory = SamplingStrategyFactory()
strategy = factory.create_strategy('uncertainty', n_partitions=3, random_state=42)

# Применяем стратегию
strategy.fit(features, target=data['target'])
partitions = strategy.get_partitions(features, target=data['target'])

# Получаем массив оценок неопределённости
uncertainty_scores = strategy.get_uncertainty_scores()

# Предсказания для подсчёта ошибок
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(features, data['target'])
predictions = model.predict(features)

for name, part in partitions.items():

    idx = part['feature'].index.to_numpy()
    if idx.size == 0:
        print(f"{name}: 0 samples (skipped)")
        continue

    y_true = np.asarray(part['target'])
    y_pred = predictions[idx]
    errors = (y_true != y_pred).sum()
    total = len(idx)
    pct = errors / total
    cm = confusion_matrix(y_true, y_pred)

    
    avg_unc = float(np.mean(uncertainty_scores[idx]))

    print(f"{name}: {errors} errors out of {total} ({pct:.2%}), avg uncertainty={avg_unc:.4f}")
    print("Confusion matrix for", name)
    print(cm)

