from examples.utils import create_distrib_dataset
from core.api.api_main import SamplingStrategyFactory

DATASET_SAMPLES = 10000
STRATEGY_TYPE = 'stratified'
STRATEGY_PARAMS = dict(n_partitions=4)

if __name__ == "__main__":
    data = create_distrib_dataset(DATASET_SAMPLES)
    # Использование фабрики для создания стратегии
    factory = SamplingStrategyFactory()
    strategy = factory.create_strategy(strategy_type=STRATEGY_TYPE, **STRATEGY_PARAMS)
    # Обучение и применение стратегии
    strategy.fit(data, target=['feature_1', 'feature_2', 'target'])
    partitions = strategy.get_partitions(data[['feature_1', 'feature_2']], target=data['target'])
    # Посмотрим, совпадают ли статистики в разделах с исходными
    strategy.check_partitions(partitions, data)
