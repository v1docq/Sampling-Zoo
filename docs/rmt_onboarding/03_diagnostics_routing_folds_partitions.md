# RMT Contraction Branch: Diagnostics, Routing, Folds And Partitions

Этот документ описывает, какие диагностические метрики фиксируются после обучения sampler-а и ensemble-а, как устроены routing/voting режимы, как делятся datasets на folds, и как формируются partitions.

## Где Лежат Результаты

Каждый запуск `run_rmt_contraction_regression_experiment` создает output directory вида:

```text
examples/benchmark/results/run_rmt_contraction_regression_<timestamp>/
```

Внутри обычно появляются:

| Artifact | Что содержит |
|---|---|
| `logs/strategy_runs.jsonl` | Структурные events от `BenchmarkLogger.log_strategy_run`. |
| `metrics/rmt_regression_runs.jsonl` | Инкрементальный append-only список run records. |
| `metrics/*.json` | Per-strategy/per-fold snapshots. |
| `ensemble_runs.csv` / `ensemble_runs.json` | Основная таблица всех runs. |
| `ensemble_summary.csv` | Aggregated summary по dataset/model/strategy. |
| `rmt_raw_runs.csv` | RMT-focused raw table. |
| `sample_efficiency_curve.csv` | Кривая sample efficiency по budget ratios. |
| `minimal_effective_budget.csv` | Минимальный budget ratio для delta thresholds. |
| `run_meta.json` | Metadata запуска: config, counts, status, timestamps, `experiment_plan`. |
| `report.md` | Markdown summary report. |
| `incremental_saver_errors.jsonl` | Ошибки snapshot hooks или saver-а, если они возникли. |

### Text2Image Prompt: Result Artifacts

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Experiment output directory figure: logs, metrics JSONL, ensemble tables, RMT report tables, metadata, markdown report, error log. Show incremental append and snapshot artifacts as separate labeled file stacks.
```

## Run Record: Основные Группы Метрик

Один run record соответствует одному leaf experiment: конкретный dataset, model, strategy, budget ratio и fold.

Типичная структура:

| Группа | Смысл |
|---|---|
| identity fields | `dataset`, `model`, `sampler`, `strategy`, `fold`, `budget_ratio`, `ensemble_method`. |
| model metrics | Для regression: `rmse`, `mse`, `r2`; для classification в общих runners могут быть accuracy/f1/roc_auc. |
| timings | `fit_time`, `inference_time`, sampler/model timings в `timings_sec`. |
| sample stats | Размер train, sample size, coverage ratio, chunk count, chunk sizes, target stats. |
| execution extra | `effective_partitions`, `chunks_percent`, `force_chunking`, `mode`, error payload. |
| sampler diagnostics | RMT-specific diagnostics из `RMTContractionTensorSampler.diagnostics_`. |
| contract snapshots | Dataset/fold/evaluation/partition/routing contracts, если record был создан через typed boundary. |
| budget policy | Как `budget_ratio` изменил partitions после sampler-а. |
| routing refinement | Optional EM retraining diagnostics: status, best iteration, improvement, stop reason, final imbalance. |

### Text2Image Prompt: Run Record Schema

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Run record schema infographic with grouped panels: identity, model metrics, timings, sample stats, sampler diagnostics, budget policy, error state. One completed fold record is shown as a compact structured object.
```

## RMT Sampler Diagnostics

После `RMTContractionTensorSampler.fit(...)` sampler сохраняет `diagnostics_`. Эти поля помогают понять, что именно сделал spectral sampler и насколько устойчиво выглядит разбиение.

| Поле | Интерпретация |
|---|---|
| `backend` | Фактически выбранный backend: `torch` или `numpy`. |
| `device` | Device tensor backend-а, например `cuda` или `cpu`. |
| `dtype` | Численный dtype. |
| `mode0_unfolding_shape` | Размер mode-0 unfolding matrix. Большая ширина увеличивает память и SVD cost. |
| `raw_encoded_feature_count` | Число признаков после preprocessing до feature cap. |
| `encoded_feature_count` | Число признаков после preprocessing и one-hot. |
| `encoded_feature_cap_applied` | Был ли применен cap до densify. |
| `n_views` | Число random contractions. |
| `n_views_requested` | Исходное значение `n_views`, например `"auto"` или integer. |
| `n_views_policy` | `static`, `coverage` или `spectrum_stability`. |
| `target_feature_coverage` | Цель coverage policy для `subsample`. |
| `estimated_feature_coverage` | Оцененное покрытие признаков выбранным числом views. |
| `spectrum_stability_tolerance` | Порог остановки spectrum-stability policy для `gaussian`. |
| `spectrum_stability_change` | Фактическое относительное изменение спектра на выбранном candidate. |
| `spectrum_stability_candidates` | Проверенные значения `n_views`. |
| `view_strategy` | `subsample` или `gaussian`. |
| `embedding_mode` | Например `sv_scaled`, если embedding перед кластеризацией масштабируется singular values. |
| `initial_rank` | Rank, рассчитанный как доля от размерности unfolding. |
| `selected_rank` | Итоговый rank после explained variance selection. |
| `rank_selection_method` | Сейчас основной метод: `explained_variance`. |
| `explained_variance_threshold` | Порог, например `0.95`. |
| `explained_variance_at_selected_rank` | Фактическая cumulative explained variance на selected rank. |
| `rank_by_null_edge` | Число компонент выше empirical null bulk edge. |
| `rank_by_stability` | Устойчивость singular-value outliers при повторных views. |
| `rank_by_subspace_stability` | Максимальный prefix rank с устойчивым left singular span. |
| `subspace_stability_status` | Состояние opt-in subspace diagnostics: `disabled`, `ok`, `partial`, `failed`. |
| `subspace_max_angle_quantile_degrees` | Quantile максимального principal angle на comparison rank. |
| `subspace_normalized_projection_distance_quantile` | Quantile нормированного расстояния проекторов. |
| `subspace_stability_frequency` | Доля view resamples, прошедших оба geometric threshold. |
| `singular_values` | Сингулярные значения до/после truncation, полезны для анализа spectral decay. |
| `leverage_entropy` | Энтропия распределения leverage scores. Низкая энтропия означает концентрацию leverage на малом числе точек. |
| `effective_sample_count` | Эффективное число точек по leverage distribution. |
| `n_partitions_requested` | Запрошенное число partitions. |
| `selected_n_partitions` | Выбранное число partitions при `partition_selection_method="auto"`. |
| `partition_selection_method` | `fixed` или `auto`. |
| `selected_cluster_algorithm` | Алгоритм, выбранный `SpectralClusterSelector`, если auto-selection включен. |
| `cluster_algorithms` | Список алгоритмов-кандидатов. |
| `cluster_selection_metric` | `silhouette` или `balanced_silhouette`. |
| `cluster_ensemble_method` | `best_score` или `weighted_vote`. |
| `partition_selection_scores` | Scores по кандидатам `k`. |
| `partition_selection_candidate_details` | Подробности candidates: algorithm, score, valid flag, components. |
| `partition_count` / `n_partitions` | Число построенных partitions. |
| `chunk_sizes` | Размеры chunks после selection/filtering. |

### Как Читать Диагностику

- Если `selected_rank` близок к `initial_rank`, спектр убывает медленно: пространство может быть сложным или threshold слишком высоким.
- Если `n_views_policy="coverage"`, проверяйте `estimated_feature_coverage`: низкое значение означает, что subsample-views не покрыли признаки достаточно широко.
- Если `n_views_policy="spectrum_stability"`, смотрите `spectrum_stability_change`: большое значение означает, что спектр еще нестабилен относительно числа views.
- Близкие `rank_by_null_edge` и `rank_by_subspace_stability` усиливают свидетельство устойчивого отделимого сигнала; большой разрыв означает чувствительность spectral directions к contractions.
- Rank-1 может быть нестабилен при стабильном rank-2 span из-за вращения близких singular vectors. Это нормальная геометрия, а не обязательно ошибка diagnostics.
- Если `leverage_entropy` очень низкая, sampler нашел небольшое число spectral-influential объектов. Это может быть полезно, но стоит проверить стабильность chunks.
- Если `chunk_sizes` сильно несбалансированы, выбранный clustering candidate может разделять данные на плотное ядро и редкие regions.
- Если `selected_n_partitions` сильно меньше `n_partitions_requested`, auto-selection решила, что дополнительные clusters ухудшают balanced objective или нарушают constraints.
- Если `encoded_feature_cap_applied=True`, downstream качество надо интерпретировать с учетом потери части one-hot признаков.

### Text2Image Prompt: RMT Diagnostics

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. RMT diagnostics dashboard figure with panels for singular value spectrum, cumulative explained variance, leverage distribution entropy, effective sample count, chunk size histogram, backend/device badge. Clean academic layout.
```

## Формирование Partitions

### Общий Путь В SamplingEnsemble

`SamplingEnsemble.prepare_data_partitions(...)` работает как pipeline:

1. определить `strategy_name`;
2. собрать kwargs для partitioner-а;
3. убрать служебные и unsupported kwargs;
4. создать partitioner;
5. вызвать подходящий fit-flow;
6. получить partitions;
7. применить `budget_ratio`, если он задан;
8. сохранить `self.partitions` и `self.partitioner`.

```mermaid
flowchart LR
    A["strategy config"] --> B["build kwargs"]
    B --> C["filter reserved / unsupported keys"]
    C --> D["create partitioner"]
    D --> E["fit partitioner"]
    E --> F["get partitions"]
    F --> G["apply budget_ratio policy"]
    G --> H["self.partitions"]
```

### RMT-Specific Partition Formation

Для `RMTContractionTensorSampler` partitions строятся так:

1. tabular features переводятся в dense numeric matrix;
2. при `n_views="auto"` выбирается число random views: coverage для `subsample`, spectrum-stability для `gaussian`;
3. random views строят mode-0 unfolding;
4. randomized SVD дает spectral embedding;
5. `embedding_mode="sv_scaled"` масштабирует `U` на `S`, чтобы clustering видел не только направление, но и spectral energy;
6. если `partition_selection_method="fixed"`, KMeans делит embedding на `n_partitions` clusters;
7. если `partition_selection_method="auto"`, `SpectralClusterSelector` сравнивает `kmeans`, `bisecting_kmeans`, `gmm`, optional `hdbscan` и выбирает partitions по `balanced_silhouette`/`weighted_vote`;
8. внутри каждого cluster выбираются строки по `selection_method`;
9. partitions получают имена `chunk_0`, `chunk_1`, ...

Выбор внутри cluster:

| `selection_method` | Поведение |
|---|---|
| `all` | Берет все объекты cluster-а. |
| `leverage` | Отбирает точки с большими leverage scores. |
| `maxvol` | Greedy-выбор точек, которые расширяют объем/разнообразие подпространства. |
| `hybrid` | Сочетает leverage и maxvol-like selection. |

### Budget Policy

В текущем основном experiment grid используется `budget_ratio`. Это глобальный post-partitioning cap:

```text
target_total_rows = round(budget_ratio * train_size)
```

Затем budget распределяется по partitions пропорционально их исходному размеру, а внутри partition выбирается local subset. Так `budget_ratio` контролирует общий размер обучающей выборки ensemble-а без смешивания с RMT-internal `chunk_fraction`.

### Text2Image Prompt: Partition Formation

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Partition formation figure: strategy config creates sampler, sampler fits sv_scaled spectral embedding, spectral cluster selector compares kmeans, bisecting kmeans, GMM and HDBSCAN candidates, balanced silhouette chooses partitions, selection method picks rows per cluster, budget_ratio trims total rows proportionally, final chunks feed model training.
```

## Разбиение На Folds

Fold logic живет в `EnsembleFoldBenchmarkExecutor`, а не в sampler-е.

### OpenML Raw Dataset

Для OpenML raw bundle используется один заранее подготовленный train/test split:

```text
OpenML dataset -> load_split_data -> FoldSplit(train, test, fold=None)
```

Такой режим нужен, чтобы не держать большой датасет в памяти несколько раз и не создавать лишние CV splits для больших задач.

### Локальный Dataset

Для локальных/обычных bundles используется KFold:

```text
dataset features/target -> KFold(cv_folds) -> FoldSplit(train_idx, test_idx, fold=i)
```

### Train/Validation Split

Для ensemble training нужен validation set, чтобы:

- оценить качество каждой chunk-модели;
- посчитать validation weights;
- сделать forward selection active models;
- контролировать early stopping.

Если train set маленький, executor может не выделять отдельный validation split для direct mode. Если включен forced chunking на малом train, создается небольшой validation split, чтобы ensemble мог корректно оценить chunk-модели.

### Text2Image Prompt: Fold Splitting

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Fold splitting figure with two lanes: OpenML raw dataset uses one predefined train/test split; local dataset uses KFold. Both lanes feed train/validation preparation and then direct or ensemble execution.
```

## Direct Model Vs Chunked Ensemble

`EnsembleFoldBenchmarkExecutor._build_execution_plan(...)` выбирает, как выполнить fold:

- `direct`: одна модель обучается на всем train set;
- `ensemble`: данные режутся на partitions, затем обучается модель на каждом chunk;
- forced modes используются для baseline или малых datasets, где нужно явно проверить chunking.

Direct baseline важен для `RMSE_ref`: он задает точку сравнения для sample efficiency curve и minimal effective budget.

### Text2Image Prompt: Execution Plan

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Execution plan decision figure: input train size and strategy config, branch to full direct model baseline or chunked ensemble, with effective partitions and budget ratio controls shown as small parameter boxes.
```

## Voting, Weighted И Routed Weighted

`SamplingEnsemble.ensemble_predict(...)` поддерживает несколько способов aggregation.

### Voting

Для regression voting - это простое среднее:

```text
y_hat(x) = (1 / K) * sum_k f_k(x)
```

Для classification voting использует class/probability aggregation в зависимости от доступного output-а модели.

### Weighted

Каждая chunk-модель получает validation weight. Для regression типично используется inverse error:

```text
w_k proportional to 1 / error_k
y_hat(x) = sum_k w_k f_k(x)
```

Weights нормализуются, чтобы сумма была равна 1.

### Routed Weighted

Routed weighted добавляет row-wise вероятность принадлежности объекта к partition:

```text
a_k(x) = routing_k(x) * validation_weight_k
w_k(x) = a_k(x) / sum_j a_j(x)
y_hat(x) = sum_k w_k(x) f_k(x)
```

Если partitioner поддерживает `predict_partition_proba`, используется она. Если нет, fallback:

1. `predict_partitions` -> one-hot routing;
2. uniform weights, если routing недоступен.

Это делает `routed_weighted` безопасным для non-routing samplers, хотя максимальный смысл он имеет для `RMTContractionTensorSampler`.

### Routed Router Modes

`RoutedWeightedRouter` отделяет routing-логику от `SamplingEnsemble`.

| Режим | Что делает |
|---|---|
| `spectral` | Default. Использует probabilities из sampler-а, обычно `RMTContractionTensorSampler.predict_partition_proba`. |
| `learned_head` | Legacy head, обучаемый на base routing features. Оставлен для сравнения и совместимости. |
| `constrained_gating` | Torch gating head, обучаемый на validation predictions с KL regularization к spectral prior и balance penalty. |

Для `constrained_gating` torch импортируется лениво, поэтому окружение без torch может импортировать benchmark код, но сам режим будет skipped или fallback-иться при отсутствии backend-а.

### Routing Diagnostics

В `validation_diagnostics.routing` и `test_routing_diagnostics` фиксируются:

| Поле | Интерпретация |
|---|---|
| `n_rows`, `n_models` | Сколько строк и активных chunk-моделей участвовало в routing. |
| `router_mode`, `router_head_status` | Какой router использовался и был ли он обучен/skipped. |
| `mean_max_probability`, `median_max_probability` | Уверенность router-а. Слишком высокие значения могут означать hard routing. |
| `mean_entropy`, `mean_normalized_entropy` | Неопределенность routing distribution. |
| `hard_assignment_counts` | Сколько строк ушло в каждый chunk по argmax. |
| `soft_assignment_mass` | Суммарная probability mass по chunks. |

## EM Routed Retraining

`routing_refinement="em_retraining"` - отдельный opt-in режим для `routed_weighted`. Он нужен, когда хочется не только смешивать уже обученные chunk-модели, но и переобучить их под фактические routed assignments.

Алгоритм:

1. E-step: на train pool считаются текущие routing probabilities.
2. Assignment: probabilities переводятся в hard top-1 assignments.
3. Guardrail: если routed partition меньше `em_min_partition_size`, итерация не принимается.
4. M-step: каждая chunk-модель переобучается на назначенном routed partition.
5. Router refresh: если `em_refit_router=True`, обновляется learned/constrained router и local validation metrics.
6. Acceptance: итерация принимается только если validation metric улучшилась не меньше `em_min_improvement`.
7. Restore: если `em_keep_best=True`, состояние ensemble возвращается к лучшему validation snapshot.

Diagnostics пишутся в `validation_diagnostics.routing_refinement`:

| Поле | Интерпретация |
|---|---|
| `status` | `disabled`, `skipped` или `completed`. |
| `initial_metric`, `best_metric`, `metric_improvement` | Validation metric до/после refinement. |
| `best_iteration` | Индекс лучшей EM-итерации; `0` означает, что лучше исходного состояния не стало. |
| `stop_reason` | `max_iterations`, `no_improvement`, `min_partition_size` или skipped reason. |
| `iterations` | Per-iteration metric, improvement, assignment-change rate, entropy before/after, imbalance. |
| `final_partition_sizes`, `final_imbalance_ratio` | Итоговые routed partition sizes и imbalance. |

### Text2Image Prompt: EM Routed Retraining

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. EM routed retraining diagnostics figure: spectral router responsibilities, hard top-1 assignment, min partition size guardrail, model refit, router refresh, validation accept or restore best, diagnostics table with improvement, entropy before and after, assignment change, imbalance.
```

### Text2Image Prompt: Routing Weights

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Routed weighted ensemble figure: a test point is projected into spectral embedding, distances to chunk centroids produce routing probabilities, validation scores produce model weights, the two are multiplied and normalized before weighted prediction aggregation.
```

## RMT Routing Математика

В RMT sampler новый объект `x` проходит тот же preprocessing и random views, что и train data. Затем:

```text
M_new = [phi_1(X_new), ..., phi_V(X_new)]
Z_new = M_new V_r S_r^{-1}
```

где `V_r` и `S_r` взяты из train SVD. Если sampler обучался с `embedding_mode="sv_scaled"`, train centroids построены в пространстве `U_r S_r`, а projection новых rows должен быть приведен к тому же embedding convention внутри backend/sampler path. Далее считаются расстояния до active centroids:

```text
d_c(x)^2 = ||Z_new(x) - mu_c||_2^2
p(c | x) = softmax_c(-d_c(x)^2 / temperature)
```

Shrinkage сглаживает routing:

```text
p_final(c | x) = (1 - shrinkage) p(c | x) + shrinkage / C
```

Параметры:

- `routing_temperature`: управляет резкостью softmax;
- `routing_shrinkage`: защищает от слишком уверенного routing-а;
- `selected_rank`: влияет на embedding и, соответственно, на расстояния;
- `partition_names_`: задает порядок columns в `predict_partition_proba`.

### Text2Image Prompt: RMT Routing Mathematics

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Mathematical routing panel: new tabular row, same random contractions, projection with V_r and S_r inverse into spectral embedding, distances to centroids, softmax temperature, shrinkage to uniform prior, probabilities aligned with partition names.
```

## Sample Efficiency И Minimal Effective Budget

`RMTReportTableBuilder` строит две ключевые производные таблицы:

### Sample Efficiency Curve

Для каждой комбинации dataset/model/sampler/ensemble method/budget ratio сравнивается RMSE с baseline:

```text
rmse_drop = (rmse - rmse_ref) / rmse_ref
```

Если `rmse_drop <= delta`, budget считается эффективным при заданном delta.

### Minimal Effective Budget

Для thresholds `delta = 1%, 3%, 5%` выбирается минимальный `budget_ratio`, который удерживает качество в допустимой зоне относительно baseline. Это отвечает на главный практический вопрос: какую долю train data нужно оставить, чтобы получить почти такое же качество.

В текущих RMT таблицах также сохраняются axes/diagnostics, важные для анализа причин качества: `view_strategy`, `n_views`, `n_views_policy`, `selected_rank`, `selected_n_partitions`, `selected_cluster_algorithm`, routing refinement status/improvement/final imbalance.

### Text2Image Prompt: Sample Efficiency

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Sample efficiency figure: budget ratio on x-axis, RMSE drop on y-axis, horizontal delta thresholds at 1, 3, 5 percent, minimal effective budget marked by vertical lines for each threshold.
```

## Практические Инварианты Для Тестов

Для поддержки этой ветки полезно проверять следующие инварианты:

- `predict_partition_proba(X).sum(axis=1) == 1` с численной tolerance;
- порядок probabilities совпадает с `partition_names_`;
- `selected_rank <= initial_rank`;
- при нулевом spectrum selected rank fallback-ится к `min_rank`;
- sparse one-hot cap применяется до densify;
- `routed_weighted` не падает на samplers без routing API;
- `budget_ratio` уменьшает общий train row count, но не создает пустые partitions без необходимости;
- каждый completed fold создает incremental JSONL record;
- failed fold возвращает structured failed record и не прерывает весь dataset run;
- report tables умеют собираться из одного record и из пустого набора records.
- `normalize_strategy_grid(...)` детерминирован и idempotent для legacy dict configs;
- `ExperimentPlan.stage_ids()` для RMT runners стабилен;
- lazy optional imports (`torch`, `tabpfn`, `tabicl`) не должны ломать import benchmark modules;
- `routing_refinement="em_retraining"` сохраняет или улучшает validation metric за счет best snapshot restore.

### Text2Image Prompt: Test Invariants

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Test invariant checklist figure for RMT benchmark: routing probabilities sum to one, selected rank bounded by initial rank, budget policy preserves nonempty chunks, incremental record saved per fold, report tables rebuild after each record.
```
