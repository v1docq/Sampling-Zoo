# Sampling Zoo

Sampling Zoo is a research-oriented Python library for partitioning and subsampling tabular datasets. It provides a common API for conventional sampling strategies and spectral methods based on random matrix theory (RMT), together with utilities for training and routing ensembles of models over the resulting data partitions.

The repository also contains the benchmark and ablation code used to evaluate RMT-based partitioning with tabular foundation models on OpenML regression datasets.

## Main features

- A unified factory API for random, stratified, model-based, clustering, geometric, temporal, and spectral sampling strategies.
- RMT contraction sampling with multi-view representations, automatic cluster selection, partition diagnostics, and routed ensembles.
- Budget-aware partitioning for experiments that train models on a controlled fraction of the original data.
- Reproducible benchmark utilities with incremental artifacts, timing and environment logs, and reusable caches for expensive preprocessing and model fitting.
- Controlled ablations for clustering in the learned embedding space versus the original feature space.

## Installation

Clone the repository and install the core package:

```bash
git clone https://github.com/v1docq/Sampling-Zoo.git
cd Sampling-Zoo
python -m pip install -e .
```

To run the benchmark suite, install the experiment dependencies:

```bash
python -m pip install -r requirements.txt
```

Some benchmark configurations use TabPFN or TabICL and are intended for a CUDA-capable environment. TabPFN may require one-time acceptance of its model license before the weights can be downloaded.

## Quick start

The factory returns fitted strategies through a consistent interface:

```python
import numpy as np
import pandas as pd

from sampling_zoo.core.api.api_main import SamplingStrategyFactory

rng = np.random.default_rng(42)
features = pd.DataFrame(rng.normal(size=(1_000, 12)))
target = pd.Series(rng.normal(size=1_000))

factory = SamplingStrategyFactory()
sampler = factory.create_and_fit(
    "random",
    data=features,
    target=target,
    strategy_kwargs={"n_partitions": 4, "random_state": 42},
)
partitions = sampler.get_partitions(features, target)

print({name: len(rows["feature"]) for name, rows in partitions.items()})
```

A complete example of RMT partitioning and routed expert training is available in [`examples/special_strategy/rmt_contraction_example.py`](examples/special_strategy/rmt_contraction_example.py):

```bash
python examples/special_strategy/rmt_contraction_example.py
```

## Experiments

The main regression benchmark entry points are:

- [`rmt_regression_medium_datasets.py`](examples/benchmark/rmt_regression_medium_datasets.py) and [`rmt_regression_big_datasets.py`](examples/benchmark/rmt_regression_big_datasets.py) for the standard OpenML experiments.
- [`embedding_space_ablation_medium_datasets.py`](examples/benchmark/embedding_space_ablation_medium_datasets.py) and [`embedding_space_ablation_big_datasets.py`](examples/benchmark/embedding_space_ablation_big_datasets.py) for comparing clustering in the RMT embedding and in the original feature space.

Each script documents its available datasets and execution profiles through `--help`, for example:

```bash
python examples/benchmark/embedding_space_ablation_medium_datasets.py --help
```

Benchmark runs write metrics, protocol metadata, environment information, timestamped console output, and structured event logs to the selected output directory. Large-dataset runs can require substantial CPU time, GPU memory, and OpenML download time.

## Repository structure

- `sampling_zoo/core/sampling_strategies/` — sampling and partitioning implementations.
- `sampling_zoo/core/utils/sampling_ensemble.py` — expert training, aggregation, and routing.
- `examples/` — API examples and experiment entry points.
- `test/` — unit and experiment-contract tests.
- `docs/experiments/` — protocols and artifacts for the extended RMT research program.

## License

Sampling Zoo is distributed under the BSD 3-Clause License.
