#  Первичный план - 04.05.26

> Это исторический общий план. Актуальная целевая проверка routing geometry и bulk/spike topology описана в [двухфазной постановке эксперимента](RMT%20routing%20geometry%20и%20bulk-spike%20experiment.md).

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

В `rmt_regression_medium_datasets.py` историческая исследовательская сетка использует
бюджеты `(0.1, 0.3, 0.5, 0.75, 0.9)`. Она сохраняется для воспроизводимости прошлых
запусков и локальных ablations.

Основная ось:

```python
DEFAULT_BUDGET_RATIOS = (0.1, 0.3, 0.5, 0.75, 0.9)
```

После завершения ablations финальный entrypoint
`rmt_regression_full_grid.py` использует основную сравнительную сетку
`(0.01, 0.05, 0.10, 0.20)`. Она строится как explicit non-Cartesian scenario grid и
сравнивает LightGBM, TabPFN in-context и TabPFN fine-tuning в independent и
concatenated постановках. Полное описание приведено в
`docs/rmt_onboarding/16_final_regression_full_grid.md`.

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
cluster_ensemble_method = "coassociation"
cluster_target_type = problem_type
missing_class_penalty_weight = 0.25
single_class_penalty_weight = 0.50
class_distribution_drift_weight = 0.25
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
6. `cluster_ensemble_method`: `coassociation` vs legacy `weighted_vote` для изоляции эффекта consensus labels.
7. `routing_refinement`: `"none"` vs `"em_retraining"`.
8. `cluster_selection_metric`: default `balanced_silhouette` vs opt-in `validation_proxy`; сравнение вести при одинаковых algorithms, budget, router и seed.

## Изолированная абляция выбора партиций

Для пункта 8 используется отдельный entrypoint
`examples/benchmark/rmt_partition_selection_ablation.py`. Он намеренно фиксирует
остальные оси эксперимента:

- strategy: `rmt_contraction`;
- ensemble: `routed_weighted`;
- router: `spectral`;
- view strategy: `gaussian`;
- model по умолчанию: `lightgbm`;
- budget grid: `(0.01, 0.03, 0.05, 0.10, 0.20)`.

Варьируется только `cluster_selection_metric`: `balanced_silhouette` и
`validation_proxy`. Каждый config name содержит явный `selection_*` tag, поэтому
incremental resume не смешивает paired runs. Помимо общих RMT-таблиц формируется
`partition_selection_comparison.csv`; delta определяется как
`validation_proxy - balanced_silhouette`, поэтому отрицательная RMSE delta означает
улучшение validation-driven policy.

# Диагностики, которые обязательно анализировать

Помимо `rmse`, `fit_time`, `inference_time`, в отчете нужно смотреть:

- `n_views`, `n_views_policy`, `spectrum_stability_change`, `estimated_feature_coverage`;
- `selected_rank`, `explained_variance_at_selected_rank`;
- `selected_n_partitions`, `selected_cluster_algorithm`, `partition_selection_scores`;
- для classification: class counts per chunk, missing-class fraction, single-class chunk/sample fraction и class-distribution drift для каждого candidate;
- для `validation_proxy`: baseline/candidate loss, relative gain, routed validation counts и fallback validation fraction;
- `chunk_size_imbalance`, target drift per chunk;
- `mean_max_probability`, routing entropy, hard assignment counts на validation/test;
- EM diagnostics: `routing_refinement_status`, `routing_refinement_metric_improvement`, `routing_refinement_final_imbalance`.
- вычислительная эффективность: `fit_rows_per_second`,
  `inference_rows_per_second`, фактическое число model-fit rows и активных моделей;
- для LightGBM: суммарные trees/leaves/splits, средняя и максимальная глубина,
  entropy gain importance и SHAP concentration.
