from core.api.api_main import SamplingStrategyFactory
from core.utils.synt_data import create_noisy_dataset

DATASET_SAMPLES = 10000
STRATEGY_TYPE = 'feature_clustering'
STRATEGY_PARAMS = dict( n_clusters=3,method='kmeans')
CLUSTERING_MODELS = ['kmeans','dbscan']
if __name__ == "__main__":
    # Создание данных
    data = create_noisy_dataset(DATASET_SAMPLES)
    result_dict = {}
    # Использование фабрики для создания стратегии
    for model in CLUSTERING_MODELS:
        factory = SamplingStrategyFactory()
        STRATEGY_PARAMS['method'] = model
        strategy = factory.create_strategy(strategy_type=STRATEGY_TYPE,**STRATEGY_PARAMS)
        # Обучение и применение стратегии
        strategy.fit(data[['feature_1', 'feature_2']], target=data['target'])
        partitions = strategy.get_partitions(data[['feature_1', 'feature_2']], target=data['target'])
        result_dict.update({model:partitions})
    _ = 1