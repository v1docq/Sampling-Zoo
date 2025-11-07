from sklearn.linear_model import LogisticRegression
from core.api.api_main import SamplingStrategyFactory
from examples.utils import create_sklearn_dataset

TASK_TYPE = 'classification'
STRATEGY_TYPE = 'difficulty'
STRATEGY_PARAMS = dict(n_partitions=3, random_state=42)
if __name__ == "__main__":
    data = create_sklearn_dataset(TASK_TYPE)
    features = data[['feature_1', 'feature_2']]
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(features, data['target'])

    # Создаём стратегию и передаём в неё модель
    STRATEGY_PARAMS['model'] = model
    factory = SamplingStrategyFactory()
    strategy = factory.create_strategy(strategy_type=STRATEGY_TYPE, **STRATEGY_PARAMS)

    # Применяем стратегию
    strategy.fit(features, target=data['target'])
    partitions = strategy.get_partitions(features, target=data['target'])

    predictions = model.predict(features)
    _ = 1