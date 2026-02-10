import numpy as np
import matplotlib.pyplot as plt
from core.api.api_main import SamplingStrategyFactory


def create_synthetic_tensor(n_frames=50, height=30, width=30, noise_level=0.1):
    """
    Создает синтетический тензор с движущимся квадратом.

    Args:
        n_frames: Количество кадров
        height: Высота кадра
        width: Ширина кадра
        noise_level: Уровень шума

    Returns:
        np.ndarray: Синтетический тензор
    """
    tensor = np.zeros((n_frames, height, width))
    for t in range(10, 40):
        h_start = int((t - 10) * 0.5) + 5
        w_start = int((t - 10) * 0.5) + 5
        tensor[t, h_start:h_start+5, w_start:w_start+5] = 10.0
    # Шум
    tensor += np.random.normal(0, noise_level, tensor.shape)

    return tensor


def temporal_sampling_example(tensor, factory):
    """
    Пример семплирования по времени (мода 0).

    Args:
        tensor: Входной тензор
        factory: Фабрика стратегий
    """
    sampler = factory.create_and_fit(
        'tensor_energy',
        tensor,
        strategy_kwargs={
            'sample_size': 100,
            'modes': [0], 
            'approx_rank': 5,
            'return_weights': True,
            'random_state': 42
        }
    )
    sampled_indices, weights = sampler.sample_indices(replace=True)

    print("\nРезультаты семплирования (индексы кадров):")
    indices_flat = [idx[0] for idx in sampled_indices]
    print(sorted(indices_flat))

    in_event = sum(10 <= idx <= 40 for idx in indices_flat)
    print(f"\nКадров из активного события: {in_event} из 100")

    plt.figure(figsize=(10, 4))
    plt.plot(sampler.joint_probs, label='Вероятность выбора (Leverage Score)')
    plt.axvspan(10, 40, color='yellow', alpha=0.3, label='Область события')
    plt.scatter(indices_flat, np.zeros_like(indices_flat), color='red', zorder=5, label='Выбранные семплы')
    plt.title("Распределение вероятностей семплирования по времени")
    plt.xlabel("Номер кадра")
    plt.ylabel("Вероятность")
    plt.legend()
    plt.tight_layout()
    plt.show()


def spatial_sampling_example(tensor, factory):
    """
    Пример пространственного семплирования (моды 1 и 2).

    Args:
        tensor: Входной тензор
        factory: Фабрика стратегий
    """

    spatial_sampler = factory.create_and_fit(
        'tensor_energy',
        tensor,
        strategy_kwargs={
            'sample_size': 150,
            'modes': [1, 2],
            'approx_rank': 3,
            'random_state': 42
        }
    )

    spatial_indices = spatial_sampler.sample_indices(replace=True)

    height, width = tensor.shape[1], tensor.shape[2]
    heatmap = np.zeros((height, width))
    for h, w in spatial_indices:
        heatmap[h, w] += 1

    plt.figure(figsize=(5, 5))
    plt.imshow(np.sum(tensor, axis=0), cmap='gray', alpha=0.5)
    plt.scatter([w for h, w in spatial_indices], [h for h, w in spatial_indices],
                c='red', s=20, label='Семплы')
    plt.title("Пространственные семплы на фоне суммы кадров")
    plt.legend()
    plt.show()


def main():
    """Основная функция примера."""
    # Создаем синтетический тензор
    tensor = create_synthetic_tensor()

    # Инициализируем фабрику
    factory = SamplingStrategyFactory()

    # Пример 1: Семплирование по времени
    temporal_sampling_example(tensor, factory)

    # Пример 2: Пространственное семплирование
    spatial_sampling_example(tensor, factory)


if __name__ == "__main__":
    main()