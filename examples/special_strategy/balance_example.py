from core.api.api_main import SamplingStrategyFactory
from core.utils.synt_data import create_noisy_dataset

DATASET_SAMPLES = 10000
STRATEGY_TYPE = 'balance'
STRATEGY_PARAMS = dict(balance_method='smote', n_partitions=4)
if __name__ == "__main__":
    data = create_noisy_dataset(DATASET_SAMPLES)
    factory = SamplingStrategyFactory()
    strategy = factory.create_and_fit(
        strategy_type=STRATEGY_TYPE,
        data=data[['feature_1', 'feature_2']],
        target=data['target'],
        strategy_kwargs=STRATEGY_PARAMS,
    )
    partitions = strategy.get_partitions()
    # 4. Анализ статистик в сбалансированных разделах
    print("\nPartition statistics after balancing with SMOTE:")
    for name, partition in partitions.items():
        feature_df, target_series = partition['feature'], partition['target']
        print(f"\n{name} ({len(target_series)} samples):")
