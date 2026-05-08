#  Первичный план - 04.05.26

1. Прогнать `rmt_contraction` против `random`, `difficulty`, `feature_clustering` на regression datasets из `AMBL_regression_suite.csv` / OpenML regression suite.
2. Разделить результаты на:
    - маленькие датасеты -  `<20k` семплов;
    - средние датасеты `20k–200k` семплов;
    - больше датасеты `>200k` семплов.
3. Для каждого датасета построить кривую зависимости качества от `budget_ratio`. `chunk_fraction` больше не является основной осью benchmark grid и должен жить в отдельной RMT ablation.
4. Сравнить два режима:
    - `ensemble_method="voting"`;
    - `ensemble_method="routed_weighted"`.

# Текущая реализация runner-а

Основной runner: `examples/benchmark/rmt_regression_medium_datasets.py`.

Важные технические решения:

1. `RMTRegressionExperimentOrchestrator.run()` исполняет типизированный `ExperimentPlan`.
2. Strategy configs сначала строятся как raw dict, затем нормализуются в `StrategyGridContract`.
3. `EnsembleChunkBenchmarkRunner` принимает и legacy dict, и typed `StrategyGridContract`.
4. Результаты пишутся инкрементально после каждого leaf-run в `metrics/rmt_regression_runs.jsonl`.
5. Загрузка OpenML dataset bundles в `_load_datasets()` отслеживается через `tqdm`: stage `Load OpenML datasets`, подэтапы `resolve suite tasks` и `apply row caps`.
6. Реальная загрузка OpenML split data остается в fold runner-е через `dataset.load_split_data(show_progress=...)`.

# Текущая RMT grid-логика

Основная ось:

```python
DEFAULT_BUDGET_RATIOS = (0.1, 0.3, 0.5, 0.75, 0.9)
```

Для `rmt_contraction` дополнительно сравниваются:

- `view_strategy`: сейчас в medium runner основной режим `gaussian`, но метод поддерживает `subsample`;
- `router`: `spectral`, `constrained_gating` для `routed_weighted`;
- `n_views="auto"`:
  - coverage policy для `subsample`;
  - spectrum-stability policy для `gaussian`.

Default RMT strategy config содержит:

```python
embedding_mode = "sv_scaled"
partition_selection_method = "auto"
cluster_algorithms = ["kmeans", "bisecting_kmeans", "gmm", "hdbscan"]
cluster_selection_metric = "balanced_silhouette"
cluster_ensemble_method = "weighted_vote"
initial_rank_fraction = 0.25
rank_selection_method = "explained_variance"
explained_variance_threshold = 0.95
```

# Optional ablations

Отдельные ablations, которые не должны смешиваться с основной budget curve:

1. `chunk_fraction`: внутренняя RMT-ручка размера cluster subset.
2. `view_strategy`: `subsample` vs `gaussian`.
3. `n_views_policy`: `coverage` vs `spectrum_stability` vs static integer.
4. `partition_selection_method`: `fixed` vs `auto`.
5. `cluster_algorithms`: исключение/добавление `gmm`, `bisecting_kmeans`, `hdbscan`.
6. `routing_refinement`: `"none"` vs `"em_retraining"`.

# Диагностики, которые обязательно анализировать

Помимо `rmse`, `fit_time`, `inference_time`, в отчете нужно смотреть:

- `n_views`, `n_views_policy`, `spectrum_stability_change`, `estimated_feature_coverage`;
- `selected_rank`, `explained_variance_at_selected_rank`;
- `selected_n_partitions`, `selected_cluster_algorithm`, `partition_selection_scores`;
- `chunk_size_imbalance`, target drift per chunk;
- `mean_max_probability`, routing entropy, hard assignment counts на validation/test;
- EM diagnostics: `routing_refinement_status`, `routing_refinement_metric_improvement`, `routing_refinement_final_imbalance`.
