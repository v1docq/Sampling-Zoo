from core.api.api_main import SamplingStrategyFactory
from core.utils.synt_data import create_noisy_dataset

DATASET_SAMPLES = 10000

if __name__ == "__main__":
    data = create_noisy_dataset(DATASET_SAMPLES)
    factory = SamplingStrategyFactory()
    partitions = factory.fit_transform(
        'random',
        data[['feature_1', 'feature_2']],
        target=data['target'],
        strategy_kwargs={'n_partitions': 3},
    )

    print("Partitions created:")
    for name, partition in partitions.items():
        print(f"{name}: {len(partition['feature'])} samples")