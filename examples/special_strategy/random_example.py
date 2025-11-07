from examples.utils import create_noisy_dataset
from core.api.api_main import SamplingStrategyFactory

DATASET_SAMPLES = 10000

if __name__ == "__main__":
    data = create_noisy_dataset(DATASET_SAMPLES)
    # Использование фабрики для создания стратегии
    factory = SamplingStrategyFactory()
    strategy = factory.create_strategy('random_split', n_partitions=3)

    # Обучение и применение стратегии
    strategy.fit(data[['feature_1', 'feature_2']], target=data['target'])
    partitions = strategy.get_partitions(data[['feature_1', 'feature_2']], target=data['target'])

    print("Partitions created:")
    for name, indices in partitions.items():
        print(f"{name}: {len(indices)} samples")