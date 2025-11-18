import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from core.api.api_main import SamplingStrategyFactory
from core.utils.synt_data import create_noisy_dataset

DATASET_SAMPLES = 10000

if __name__ == "__main__":
    data = create_noisy_dataset(DATASET_SAMPLES)
    features = data[['feature_1', 'feature_2']]

    # Создаём стратегию UncertaintySampler
    factory = SamplingStrategyFactory()
    strategy = factory.create_and_fit(
        'uncertainty',
        data=features,
        target=data['target'],
        strategy_kwargs={'n_partitions': 3, 'random_state': 42},
    )
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

