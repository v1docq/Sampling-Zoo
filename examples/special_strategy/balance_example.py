from core.api.api_main import SamplingStrategyFactory
from examples.utils import create_noisy_dataset

DATASET_SAMPLES = 10000
STRATEGY_TYPE = 'balance'
STRATEGY_PARAMS = dict(balance_method='smote', n_partitions=4)
if __name__ == "__main__":
    data = create_noisy_dataset(DATASET_SAMPLES)
    factory = SamplingStrategyFactory()
    strategy = factory.create_strategy(strategy_type=STRATEGY_TYPE, **STRATEGY_PARAMS)
    # Обучаем сэмплер
    strategy.fit(data[['feature_1', 'feature_2']], data['target'])
    partitions = strategy.get_partitions()
    # 4. Анализ статистик в сбалансированных разделах
    print("\nPartition statistics after balancing with SMOTE:")
    for name, (feature_df, target_series) in partitions.items():
        print(f"\n{name} ({len(target_series)} samples):")
