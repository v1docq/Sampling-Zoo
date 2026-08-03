# RMT Contraction Branch: Class And Method Reference

Этот документ описывает ключевые классы ветки RMT contraction: зачем нужен каждый класс, какие методы являются публичными, какие внутренние методы реализуют отдельные этапы поведения, и где находятся математические алгоритмы.

Основной принцип чтения кода: публичный метод класса должен быть коротким pipeline-методом, а сложное поведение должно быть вынесено во внутренние методы с одним смыслом. Для spectral/tensor samplers численные kernels должны жить в backend-слое, а sampler должен оставаться orchestrator-ом sampling strategy.

## RMTRegressionExperimentOrchestrator

Файл: `examples/benchmark/run_rmt_contraction_regression_experiment.py`

Назначение: верхнеуровневый orchestrator regression benchmark-а. Он не обучает модели сам и не строит RMT partitions. Он связывает config, logger, incremental saver, dataset loading, strategy configs, runner и report artifacts.

### Основные Методы

| Метод | Назначение |
|---|---|
| `run()` | Публичный pipeline запуска. Строит `ExperimentPlan`, передает его в `_execute_experiment_plan`, при исключении помечает incremental saver как failed. |
| `_build_experiment_plan()` | Создает типизированный план стадий через `build_standard_rmt_experiment_plan(...)` и сохраняет его для metadata. |
| `_execute_experiment_plan(plan)` | Последовательно исполняет stage ids: runtime, logger, runner, datasets, strategy grid, datasets run, reports, metadata, finalize. |
| `_prepare_runtime()` | Настраивает runtime-предусловия, например фильтры warning-ов. |
| `_create_logger()` | Создает `BenchmarkLogger` для output directory текущего запуска. |
| `_create_incremental_recorder(logger)` | Возвращает callback, который будет вызван после каждого run record. |
| `_create_incremental_saver(logger)` | Создает `IncrementalExperimentSaver` и регистрирует snapshot hooks для промежуточных отчетов. |
| `_create_runner(logger)` | Создает `EnsembleChunkBenchmarkRunner`, передавая ему logger и `on_record` callback. |
| `_load_datasets()` | Загружает доступные regression datasets с учетом row cap и списка задач; OpenML discovery и cap/wrapping показываются через `tqdm`. |
| `_load_available_datasets()` | Внутренний helper для перебора dataset specs и graceful skip недоступных datasets. |
| `_build_strategy_grid()` | Создает raw strategy configs и нормализует их в `StrategyGridContract`. |
| `_run_experiment(datasets, strategy_configs, runner, logger)` | Итерация по datasets; для каждого dataset запускает сетку моделей и стратегий через runner. |
| `_build_report_artifacts(run_records, logger)` | Строит summary/report artifacts. Если активен incremental saver, делегирует snapshot rebuild. |
| `_build_run_meta(...)` | Формирует metadata payload: config, counts, timestamps, status. |
| `_write_run_metadata(logger, run_records)` | Финализирует metadata через saver или пишет metadata напрямую. |
| `_announce_completion(logger)` | Возвращает путь к output directory и печатает/логирует завершение. |

### Почему Это Важно

`run()` - пример “thin shell” паттерна: он описывает порядок операций, но не содержит вложенных циклов и специальных случаев. Если нужно менять поведение загрузки данных, сохранения или отчетов, изменение должно попасть в отдельный метод.

## Experiment Contracts, Stages And Morphisms

Файлы:

- `sampling_zoo/core/experiment/contracts.py`
- `sampling_zoo/core/experiment/stages.py`
- `sampling_zoo/core/experiment/morphisms.py`
- `sampling_zoo/core/experiment/errors.py`

Назначение: общий framework-level слой для staged experiments. Он делает boundary между raw runtime objects и устойчивыми typed snapshots: dataset, strategy grid, model, fold, partitions, chunk models, routing, evaluation, run records и artifacts.

### Ключевые Классы И Функции

| Объект | Назначение |
|---|---|
| `ExperimentPlan` | Immutable последовательность `StageRequest`, которую исполняет orchestrator. |
| `ExperimentStageId` | Enum стандартных стадий: `prepare_runtime`, `create_logger`, `create_runner`, `load_datasets`, `build_strategy_grid`, `run_datasets`, `build_reports`, `write_metadata`, `finalize`. |
| `StageRequest` / `StageResult` | Типизированные вход/выход stage handler-а. |
| `StrategySpec` / `StrategyGridContract` | Типизированная strategy grid. Raw dict configs материализуются обратно только на legacy boundary. |
| `DatasetContract`, `FoldContract`, `PartitionContract`, `RoutingContract`, `EvaluationContract` | Compact snapshots для логирования, диагностики и invariant tests. |
| `PartitionTrainingRequest` / `PartitionTrainingResult` | Internal contract вокруг `SamplingEnsemble.train_partition_models(...)`. |
| `normalize_strategy_grid(...)` | Pure morphism: raw mapping configs -> `StrategyGridContract`. |
| `materialize_strategy_grid(...)` | Pure morphism: typed grid -> legacy dict kwargs. |
| `build_standard_rmt_experiment_plan(...)` | Pure builder стандартного RMT stage plan. |

### Почему Это Важно

Новые runner-ы должны идти по каноническому пути:

```text
raw config -> typed validation/spec -> pure morphism -> runtime shell -> artifacts
```

Так проще проверять invariants: порядок stage plan стабилен, нормализация configs детерминирована, а records можно сравнивать без чтения внутренних объектов sklearn/torch.

### Text2Image Prompt: Orchestrator

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Class responsibility figure for RMTRegressionExperimentOrchestrator: a thin public run method calling private stage methods. Show logger, saver, datasets, strategy configs, runner, reports as separate boxes connected by arrows.
```

## IncrementalExperimentSaver

Файл: `examples/benchmark/benchmark_incremental.py`

Назначение: устойчивое сохранение результатов длинного эксперимента. Каждый record дописывается в JSONL сразу после завершения leaf-run. Snapshot hooks перестраивают промежуточные CSV/JSON/MD artifacts.

### Основные Методы

| Метод | Назначение |
|---|---|
| `start()` | Пишет metadata со status `running`. |
| `record(record)` | Добавляет record в память, append-ит JSONL, запускает snapshot hooks. |
| `persist_snapshot(records=None)` | Перестраивает snapshot artifacts по текущим или переданным records. |
| `finalize(records=None)` | Делает финальный snapshot и пишет status `completed`. |
| `mark_failed(error)` | Пишет error event и status `failed`, не удаляя уже сохраненные records. |
| `write_metadata(status)` | Atomic write metadata JSON. |
| `_append_record_jsonl(record)` | Durable append с JSON serialization и flush/fsync. |
| `_run_snapshot_hook(hook)` | Изолированный вызов hook-а, чтобы ошибка в отчете не ломала запись record-а. |
| `_append_error(event, error)` | Логирует ошибки saver-а в отдельный JSONL. |
| `_atomic_write_json(path, payload)` | Записывает JSON через временный файл и replace. |

### Text2Image Prompt: Incremental Saver

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Durable experiment saver diagram with append-only JSONL, in-memory records, snapshot hooks, metadata states running/completed/failed, and error isolation.
```

## EnsembleChunkBenchmarkRunner

Файл: `examples/benchmark/benchmark_runner.py`

Назначение: запускает benchmark для одного dataset по сетке моделей и стратегий. После refactoring runner не владеет fold-level логикой: это делегировано `EnsembleFoldBenchmarkExecutor`.

### Основные Методы

| Метод | Назначение |
|---|---|
| `run_dataset(dataset, strategy_configs, models=None)` | Публичный метод для одного dataset. Принимает legacy dict или `StrategyGridContract`, нормализует grid, загружает OpenML split при необходимости, запускает grid, освобождает OpenML split data. |
| `_load_openml_split(dataset)` | Загружает raw train/test split для OpenML bundle один раз на dataset. |
| `_run_model_strategy_grid(...)` | Внешний цикл по моделям. |
| `_iter_models(models)` | Нормализует список model keys и model factories. |
| `_run_strategies_for_model(...)` | Цикл по strategy configs для конкретной модели. |
| `_iter_strategies(strategy_configs)` | Материализует typed grid в legacy strategy-name/config пары только на границе runner/factory. |
| `_record_run(records, record)` | Добавляет record в локальный список и вызывает `on_record`, если он задан. |
| `_release_openml_split(openml_split_data)` | Освобождает большой split object после dataset. |

### Практический Контракт

`EnsembleChunkBenchmarkRunner` должен оставаться “grid runner”. Если нужно менять fold split, validation split, direct/ensemble behavior или fold-level error handling, править нужно `EnsembleFoldBenchmarkExecutor`, а не `run_dataset`.

### Text2Image Prompt: Benchmark Runner

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Grid runner diagram for EnsembleChunkBenchmarkRunner: one dataset enters, model loop and strategy loop create fold execution requests, each request returns records to an incremental callback.
```

## EnsembleFoldBenchmarkExecutor

Файл: `examples/benchmark/benchmark_runner.py`

Назначение: отдельный класс для fold-level поведения. Он решает, как получить folds, как выделить validation, нужно ли обучать direct model или chunked ensemble, как логировать успехи и ошибки.

### Основные Методы

| Метод | Назначение |
|---|---|
| `run_strategy_folds(...)` | Публичный fold pipeline для одной пары model/strategy. |
| `split_count(dataset)` | Для OpenML raw bundle возвращает 1, для локального dataset - `cv_folds`. |
| `iter_folds(dataset, openml_split_data)` | Генерирует `FoldSplit`: OpenML holdout или KFold. |
| `run_single_fold(...)` | Выполняет один fold и возвращает record. |
| `_openml_fold(...)` | Создает fold из готового OpenML split. |
| `_local_cv_fold(...)` | Создает fold через sklearn KFold. |
| `_prepare_train_val_for_execution(...)` | Выделяет validation subset и применяет caps. |
| `_build_execution_plan(partitioner_config, train_size)` | Решает direct model vs ensemble, effective partitions, chunks_percent. |
| `_ensure_validation_split(...)` | Для forced chunking на малых данных создает validation split. |
| `_split_small_train_val(...)` | Делит небольшой train на train/val без тяжелой логики AMLB. |
| `_run_direct_model_fold(...)` | Обучает одну модель на всем train и считает predictions. |
| `_run_ensemble_fold(...)` | Создает `SamplingEnsemble`, готовит partitions, обучает chunk models, запускает inference. |
| `_log_direct_model_fold(...)` | Формирует record для direct baseline. |
| `_log_ensemble_fold(...)` | Формирует record для chunked ensemble, включая sampler diagnostics. |
| `_log_failed_fold(...)` | Возвращает structured failed record вместо потери всего запуска. |

### FoldExecutionPlan

`FoldExecutionPlan` - typed snapshot решения для fold-а:

- `mode`: direct или ensemble;
- `effective_partitions`: сколько partitions запросить;
- `chunks_percent`: какой процент chunks оставить, если strategy это поддерживает;
- flags forced behavior.

### Text2Image Prompt: Fold Executor

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Fold executor decision tree: dataset split, train validation split, execution plan, direct full model branch, chunked ensemble branch, success record, failed record. Use small labeled nodes and precise arrows.
```

## SpecialStrategyBenchmarkRunner

Файл: `examples/benchmark/benchmark_runner.py`

Назначение: legacy runner для специальных sampling strategies, которые возвращают sampled indices/scores напрямую, а не через `SamplingEnsemble` partitions. Он полезен как совместимость со старыми benchmark functions.

### Основные Методы

| Метод | Назначение |
|---|---|
| `run(...)` | Запускает набор специальных strategies на dataset. |
| `_run_single_strategy(...)` | Выполняет одну strategy, получает sampled rows, обучает model, считает metrics. |
| `_resolve_sample_indices(...)` | Достает sample indices из strategy output. |
| `_resolve_score_values(...)` | Достает score values, если strategy их вернула. |
| `_collect_metrics(...)` | Сводит model metrics, timings, sample stats и extra payload. |

### Когда Использовать

Если strategy уже умеет вернуть `sampled_indices`, но не является partitioner-ом, можно временно подключить ее через `SpecialStrategyBenchmarkRunner`. Для новых chunking strategies предпочтительнее путь через `SamplingEnsemble`.

### Text2Image Prompt: Special Strategy Runner

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Legacy special strategy runner flow: strategy output with sampled indices and scores, train one model, evaluate metrics, log run record. Contrast it with partition-based ensemble path in a small side panel.
```

## SamplingEnsemble

Файл: `sampling_zoo/core/utils/sampling_ensemble.py`

Назначение: связывает partitioner/sampler и predictive model. Он готовит partitions, применяет budget policy, обучает отдельные модели на chunks, выбирает active models и агрегирует predictions.

### Partitioning Методы

| Метод | Назначение |
|---|---|
| `prepare_data_partitions(features, target, random_state)` | Публичный pipeline подготовки partitions. |
| `_strategy_name()` | Возвращает имя strategy из config. |
| `_reserved_partitioner_config_keys()` | Список служебных ключей, которые нельзя передавать в constructor sampler-а. |
| `_build_partitioner_kwargs(...)` | Собирает kwargs для partitioner-а. |
| `_filter_partitioner_kwargs(...)` | Убирает unsupported kwargs для конкретной strategy. |
| `_with_supervised_partitioner_kwargs(...)` | Добавляет supervised helpers, если strategy требует target/support model. |
| `_create_support_model(...)` | Создает вспомогательную модель для supervised sampling. |
| `_create_partitioner(...)` | Factory для sampler/partitioner class. |
| `_fit_and_collect_partitions(...)` | Вызывает правильный fit-flow и возвращает partitions. |
| `_fit_target_aware_partitioner(...)` | Передает `target` в `RMTContractionTensorSampler.fit(...)`, чтобы auto-selection могла оценить class coverage или regression target contrast. Для остальных unsupervised partitioners прежний fit-flow не меняется. |
| `_apply_budget_policy_to_partitions(...)` | Сжимает partitions по `budget_ratio` после partitioning. |

### Training Методы

| Метод | Назначение |
|---|---|
| `train_partition_models(...)` | Публичный pipeline обучения chunk-моделей. |
| `_load_or_prepare_partitions(...)` | Использует уже готовые partitions или загружает их с диска. |
| `_train_partition_loop(...)` | Последовательно обучает модели по partition-ам. |
| `_train_partition_and_score_ensemble(...)` | Обучает одну partition-модель и обновляет ensemble score. |
| `_train_single_partition_model(...)` | Создает и fit-ит модель на одном chunk. |
| `_ensure_partition_class_coverage(...)` | Для classification добавляет недостающие классы, если нужно. |
| `_build_partition_model_info(...)` | Собирает metadata модели chunk-а. |
| `_register_partition_model(...)` | Добавляет trained model в registry ensemble-а. |
| `_evaluate_current_ensemble(...)` | Считает validation metrics текущего ensemble. |
| `_should_stop_partition_training(...)` | Early stopping по degradation rounds. |
| `_build_partition_training_request(...)` | Создает typed request snapshot для обучения chunk-моделей. |
| `_build_partition_training_result(...)` | Создает typed result snapshot: partitions, chunk models, routing contract, validation diagnostics. |
| `_run_routing_refinement(...)` | Запускает optional `RoutedEMModelRefiner`, если `routing_refinement="em_retraining"`. |
| `_finalize_partition_training(...)` | Завершает обучение и выбирает active model subset. |
| `select_best_models_forward(...)` | Forward selection: добавляет модели, которые улучшают validation metric. |

### Режим обучения partition-моделей

`SamplingEnsemble` поддерживает два явных режима через
`partition_model_mode`:

- `independent` сохраняет основной MoE-сценарий: отдельная модель обучается
  на каждом sampled partition;
- `concatenated` строит одну выборку из объединения всех sampled partitions и
  обучает одну модель при том же суммарном бюджете строк.

`concatenated` нужен как контроль эксперимента: он отделяет эффект выбора строк
RMT-семплером от эффекта разбиения на экспертов и routing. Исходные partitions
остаются в `PartitionTrainingResult`, а `PartitionTrainingRequest` отдельно
фиксирует `n_partitions`, `n_training_partitions` и `partition_model_mode`.
Комбинация `concatenated + routed_weighted` запрещена: при единственной модели
роутинг вырождается в постоянный вес 1, поэтому для контроля используется
`ensemble_method="voting"`.

### Inference Методы

| Метод | Назначение |
|---|---|
| `_run_inference(...)` | Унифицирует `predict`, `predict_proba` и regression/classification output. |
| `_validation_weights(active_models)` | Вес chunk-моделей по validation quality. |
| `_routing_weights(features, active_models)` | Row-wise routing weights через `RoutedWeightedRouter`: spectral probabilities, learned head, constrained gating или fallback. |
| `ensemble_predict(features, stage="inference", models=None)` | Главный inference method: `voting`, `weighted`, `routed_weighted`. |
| `ensemble_predict_batch(...)` | Batch inference для больших test sets. |

### Математика Ensemble

Для regression:

- `voting`: среднее prediction-ов chunk-моделей;
- `weighted`: сумма predictions с весами по validation quality;
- `routed_weighted`: row-wise веса `routing_weight(x, chunk) * validation_weight(chunk)`, затем нормализация.

Если sampler не умеет `predict_partition_proba`, routing fallback-ится к `predict_partitions`, а затем к uniform weights.

`routing_refinement="em_retraining"` является explicit opt-in режимом. По умолчанию router остается spectral, а EM refiner не запускается. Когда режим включен, `RoutedEMModelRefiner` чередует hard top-1 assignment train rows по текущему router-у и переобучение chunk-моделей, принимает итерацию только при улучшении validation metric и при `em_keep_best=True` откатывает состояние к лучшему snapshot.

### Text2Image Prompt: Sampling Ensemble

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. SamplingEnsemble diagram: partitioner creates chunks, budget policy trims chunks, one model is trained per chunk, validation scores create model weights, routing probabilities create row-wise weights, predictions are aggregated.
```

## RoutedWeightedRouter

Файл: `sampling_zoo/core/utils/ensemble_routing.py`

Назначение: отдельный объект, владеющий routing weights и routing diagnostics для `routed_weighted`. Он отделяет маршрутизацию от `SamplingEnsemble`, чтобы ensemble не разрастался логикой spectral routing, learned heads и constrained gating.

### Режимы

| Режим | Смысл |
|---|---|
| `spectral` | Default: использует `partitioner.predict_partition_proba(...)`, затем fallback к one-hot/uniform. |
| `learned_head` | Legacy RandomForest head по base routing features. Сохранен для совместимости. |
| `constrained_gating` | Torch gating head, обучаемый на validation predictions с KL regularization к spectral prior и balance penalty. |

`torch` для constrained gating импортируется лениво через `_load_torch_backend()`, поэтому импорт benchmark-кода не должен падать в окружении без torch.

### Text2Image Prompt: Routed Router

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. RoutedWeightedRouter figure: base spectral probabilities from partitioner, optional learned head, optional constrained gating neural head, validation model weights, final row-wise routing weights and diagnostics panel.
```

## RoutedEMModelRefiner

Файл: `sampling_zoo/core/utils/routed_em_refiner.py`

Назначение: explicit opt-in collaborator для совместной донастройки chunk-моделей в режиме `routed_weighted`.

### Конфигурация

| Параметр | Смысл |
|---|---|
| `routing_refinement` | `"none"` или `"em_retraining"`. Default: `"none"`. |
| `em_max_iterations` | Максимум EM-итераций. |
| `em_min_improvement` | Минимальное улучшение validation metric для принятия итерации. |
| `em_assignment_policy` | Сейчас поддерживается `hard_top1`. |
| `em_min_partition_size` | Guardrail от слишком маленьких routed partitions. |
| `em_refit_router` | Нужно ли обновлять router после M-step. |
| `em_keep_best` | Откатывать ensemble к лучшему validation snapshot. |

### Алгоритм

1. E-step: получить train-row responsibilities через текущий router.
2. Assignment: превратить responsibilities в hard top-1 partitions.
3. Guardrail: остановиться, если хотя бы один routed partition меньше `em_min_partition_size`.
4. M-step: переобучить по одной chunk-модели на каждом routed partition.
5. Router refresh: при `em_refit_router=True` обновить learned/constrained router.
6. Acceptance: принять итерацию только если validation metric улучшилась не меньше `em_min_improvement`.
7. Restore: при `em_keep_best=True` вернуть лучший snapshot.

### Text2Image Prompt: EM Routed Retraining

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. EM routed retraining figure: current router assigns train rows to chunk experts, hard top-1 assignment creates routed partitions, chunk models are refit, router is refreshed, validation metric decides accept or restore best. Include diagnostics: improvement, assignment change rate, entropy before and after, imbalance.
```

## BaseSampler

Файл: `sampling_zoo/core/sampling_strategies/base_sampler.py`

Назначение: общий base class для samplers. В RMT ветке особенно важен его preprocessing слой.

### Важные Методы

| Метод | Назначение |
|---|---|
| `_validate_positive_int`, `_validate_fraction`, `_validate_percent`, `_validate_choice` | Единые validators для config values. |
| `_configure_tabular_preprocessing(...)` | Сохраняет настройки tabular preprocessing. |
| `_fit_transform_features(data)` | Fit-transform tabular features. |
| `_fit_transform_dataframe(data)` | Numeric impute/scale + categorical one-hot. |
| `_cap_and_densify_encoded_features(X)` | Важно: sparse one-hot сначала ограничивается по `max_encoded_features`, и только потом densify. |
| `_transform_features(data)` | Transform новых features во время inference/routing. |
| `_resolve_backend()` | Выбирает `torch` или `numpy` backend. |
| `_resolve_torch_device()` | Выбирает torch device, включая CUDA availability. |
| `_to_torch_matrix(X)` | Переводит dense numpy matrix в torch tensor. |
| `_load_torch_backend()` | Модульный lazy helper: torch импортируется только при реальной необходимости, поэтому import sampler-а не ломается в окружениях без torch. |

### Text2Image Prompt: Tabular Preprocessing

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Tabular preprocessing figure: numeric columns imputed and scaled, categorical columns one-hot encoded sparsely, feature cap applied before densify, dense float matrix emitted to sampler/backend.
```

## SpectralSamplerBase

Файл: `sampling_zoo/core/sampling_strategies/spectral/base_sampler.py`

Назначение: common base для spectral/tensor samplers. Он хранит shared state и валидирует параметры, которые типичны для spectral sampling: partitions, views, chunk fractions, backend, dtype, routing controls.

### Важные Методы

| Метод | Назначение |
|---|---|
| `__init__(...)` | Валидирует общие spectral config fields и вызывает preprocessing config. |
| `_init_spectral_state()` | Инициализирует placeholders: embedding, scores, partitions, diagnostics, backend state. |
| `fit`, `build_spectral_representation`, `compute_sampling_scores`, `sample_indices`, `get_partitions` | Abstract/contract-like methods для наследников. |

### Text2Image Prompt: Spectral Base

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Base class inheritance diagram: BaseSampler provides preprocessing and backend resolution, SpectralSamplerBase adds spectral config and state, RMTContractionTensorSampler adds RMT orchestration.
```

## RMTContractionConfig

Файл: `sampling_zoo/core/sampling_strategies/spectral/rmt_contraction_sampler.py`

Назначение: immutable config object для RMT contraction sampler. Он заменяет перегруженный constructor большим количеством позиционных параметров.

### Ключевые Параметры

| Параметр | Смысл |
|---|---|
| `n_partitions` | Число кластеров/partitions до возможной фильтрации chunk budget. |
| `partition_selection_method` | `fixed` или `auto`. В `auto` число partitions выбирается через `SpectralClusterSelector`. |
| `cluster_algorithms` | Кандидаты кластеризации: `kmeans`, `bisecting_kmeans`, `gmm`, optional `hdbscan`. |
| `cluster_selection_metric` | Метрика выбора candidates: `silhouette` или `balanced_silhouette`. |
| `cluster_ensemble_method` | `best_score`, legacy `weighted_vote` или `coassociation`, который строит consensus labels из всех candidates. |
| `min_partitions`, `max_partitions` | Диапазон числа кластеров для auto-selection. |
| `max_cluster_imbalance_ratio`, `min_cluster_fraction` | Hard constraints против слишком несбалансированных или слишком маленьких clusters. |
| `cluster_target_type` | `auto`, `regression` или `classification`. Benchmark factory передает тип задачи явно; это особенно важно для целочисленной regression target, которую автоматический inference может принять за multiclass. |
| `missing_class_penalty_weight` | Вес мягкого штрафа за отсутствующие пары chunk/class. |
| `single_class_penalty_weight` | Вес дополнительного мягкого штрафа за долю одно-классовых chunks; сам факт такого chunk также является hard constraint violation. |
| `class_distribution_drift_weight` | Вес среднего, взвешенного по размеру chunk, total-variation drift между локальным и глобальным распределениями классов. |
| `validation_proxy_fraction` | Доля outer-train fold, выделяемая во внутренний deterministic holdout при `cluster_selection_metric="validation_proxy"`. |
| `validation_proxy_min_partition_rows` | Минимум proxy-train строк в partition; меньший local expert заменяется global baseline. |
| `validation_proxy_smoothing` | Аддитивное сглаживание class probabilities для classification proxy. |
| `n_views` | Сколько random feature views строить. Может быть integer или `"auto"`. |
| `n_views_policy` | `auto`, `coverage` или `spectrum_stability`. Для `subsample` auto -> coverage, для `gaussian` auto -> spectrum stability. |
| `min_views`, `max_views` | Границы автоматического выбора числа views. |
| `target_feature_coverage` | Целевое покрытие признаков для coverage policy. |
| `spectrum_stability_tolerance` | Порог изменения спектра для остановки gaussian spectrum-stability policy. |
| `embedding_mode` | `sv_scaled` масштабирует left singular vectors на singular values перед кластеризацией. |
| `view_size` | Сколько исходных признаков использовать в одном subsample-view. |
| `projection_dim` | Размерность random projection внутри view. |
| `initial_rank_fraction` | Начальный rank как доля от `min(n_samples, n_unfolding_features)`. Default: `0.25`. |
| `rank_selection_method` | Метод выбора итогового rank. Сейчас основной default: `explained_variance`. |
| `explained_variance_threshold` | Порог cumulative spectral energy. Default: `0.95`. |
| `min_rank` | Нижняя граница selected rank. |
| `null_diagnostic_enabled`, `null_*` | Opt-in empirical null references и bulk-edge diagnostics; подробно в `08_null_calibrated_spectral_diagnostics.md`. |
| `subspace_diagnostic_enabled`, `subspace_*` | Opt-in principal-angle и projection-distance diagnostics; подробно в `09_spectral_subspace_stability.md`. |
| `view_strategy` | `subsample` или `gaussian`. |
| `selection_method` | Как выбирать точки внутри cluster: `all`, `leverage`, `capped_leverage`, `maxvol`, `hybrid`. |
| `leverage_cap_quantile` | Верхний квантиль локальных leverage scores для `capped_leverage`; default `0.95`. |
| `routing_temperature` | Температура softmax routing-а. Больше значение делает weights более равномерными. |
| `routing_shrinkage` | Смешивание routing probabilities с uniform prior. |
| `backend` | `auto`, `torch`, `numpy`. |
| `device`, `dtype` | Tensor backend device и dtype. |
| `max_unfolding_elements` | Guardrail от слишком большого mode-0 unfolding. |

`approx_rank` намеренно удален из RMT config. Если он передан, sampler должен выбросить понятный `ValueError` с подсказкой использовать `initial_rank_fraction`, `rank_selection_method` и `explained_variance_threshold`.

### Text2Image Prompt: RMT Config

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Configuration schema figure for RMTContractionConfig with grouped fields: views, adaptive rank, chunking, routing, preprocessing, backend. Show removed approx_rank replaced by adaptive rank policy.
```

## RMTContractionTensorSampler

Файл: `sampling_zoo/core/sampling_strategies/spectral/rmt_contraction_sampler.py`

Назначение: partitioner, который строит chunks на табличных данных через random feature contractions, mode-0 unfolding, randomized SVD, leverage scores и clustering.

### Constructor И Config Parsing

| Метод | Назначение |
|---|---|
| `__init__(config=None, *args, **kwargs)` | Создает sampler из `RMTContractionConfig`, overrides или legacy positional inputs. |
| `_normalize_config_inputs(config, args, kwargs)` | Изолирует config parsing, kwargs overrides и removed params checks. |

Важно: constructor не должен превращаться в место сложной логики. Любые deprecated/removed params и conversion rules должны жить в `_normalize_config_inputs`.

### Fit Pipeline

| Метод | Назначение |
|---|---|
| `fit(X, y=None)` | Публичный pipeline: backend, preprocessing, unfolding, spectral basis, clusters/partitions, diagnostics. |
| `_start_fit()` | Создает RNG, resolved backend и backend object. |
| `_make_rmt_backend()` | Создает `TensorRMTBackend` или `MatrixRMTBackend`. |
| `_get_rmt_backend()` | Возвращает backend и проверяет, что он создан. |
| `_build_fit_unfolding(X_num, rng)` | Строит mode-0 unfolding на train features. |
| `_fit_spectral_basis(M)` | Вычисляет spectral basis и adaptive rank. |
| `_fit_spectral_null_diagnostic(X_num)` | Строит empirical null references через активный backend, если diagnostics включены. |
| `_fit_spectral_subspace_diagnostic(X_num)` | Оценивает устойчивость left singular subspace при повторной генерации views. |
| `_fit_clusters_and_partitions(scores, target)` | Делегирует выбор labels фиксированному KMeans или `SpectralClusterSelector`, затем строит partitions. |
| `_fit_auto_partition_clusters(embedding, target)` | Выбирает алгоритм и число clusters через auto-selection. |
| `_partition_info_from_selection(result)` | Превращает результат `SpectralClusterSelector` в diagnostics-friendly `PartitionSelectionInfo`. |
| `_build_diagnostics(M, rank_info)` | Сохраняет RMT diagnostics. |

### Математика: Random Views И Mode-0 Unfolding

Для каждого view sampler выбирает подмножество или random projection признаков. Затем backend строит contraction:

```text
M = [phi_1(X), phi_2(X), ..., phi_V(X)]
```

где `M` - mode-0 unfolding matrix размера `n_samples x unfolding_width`, а `phi_v` - view-specific feature contraction. В табличном случае это не физический тензор на диске, а матричное представление набора random tensor-like views.

| Метод | Назначение |
|---|---|
| `_make_view_specs(n_features, rng)` | Генерирует `ViewSpec` для каждого random view. |
| `_resolve_n_views_policy()` | Выбирает `coverage` для `subsample` и `spectrum_stability` для `gaussian`, если `n_views_policy="auto"`. |
| `_resolve_coverage_n_views(n_samples, n_features)` | Подбирает число views по целевому покрытию признаков и guardrail `max_unfolding_elements`. |
| `_build_spectrum_stable_fit_unfolding(X_num, rng)` | Перебирает candidates `min_views, 2*min_views, ... max_views` и останавливается, когда относительное изменение спектра меньше tolerance. |
| `_resolve_view_size(n_features)` | Выбирает размер view с учетом числа признаков. |
| `_check_unfolding_size(n_samples, n_features)` | Guardrail по `max_unfolding_elements`. |
| `_build_mode0_unfolding(X_num, view_specs)` | Делегирует backend-у построение unfolding. |

### Математика: Randomized SVD И Adaptive Rank

Sampler сначала выбирает `initial_rank`:

```text
initial_rank = ceil(initial_rank_fraction * min(n_samples, n_unfolding_features))
```

Затем backend считает truncated randomized SVD:

```text
M approx U S V^T
```

После этого sampler выбирает минимальный `selected_rank`, при котором cumulative explained variance достигает `explained_variance_threshold`:

```text
EV(k) = sum_{i=1..k} S_i^2 / sum_{i=1..initial_rank} S_i^2
selected_rank = min k such that EV(k) >= threshold
```

Если спектр нулевой или численно некорректный, используется fallback к `min_rank`.

| Метод | Назначение |
|---|---|
| `_resolve_initial_rank(n_samples, n_features)` | Вычисляет initial rank из доли пространства. |
| `_select_rank_from_spectrum(singular_values)` | Выбирает selected rank по explained variance. |
| `_store_spectral_basis(basis, rank_info)` | Truncate-ит basis и сохраняет embedding, singular values, leverage scores. |

### Математика: Leverage И Selection

Leverage score строки:

```text
l_i = ||U_r[i, :]||_2^2 / sum_j ||U_r[j, :]||_2^2
```

Он показывает, насколько объект представлен в ведущем spectral subspace. Внутри cluster-а sampler может выбирать строки по leverage, capped leverage, maxvol или hybrid.

| Метод | Назначение |
|---|---|
| `_build_partitions_from_labels(labels, rng)` | Преобразует cluster labels в partitions. |
| `_select_from_cluster(cluster_idx, scores)` | Выбирает rows внутри cluster по `selection_method`; stochastic policies получают воспроизводимый generator. |
| `_greedy_maxvol_indices(candidate_idx, target_size)` | Greedy approximation max-volume selection. |
| `_orthonormal_basis(rows)` | Вспомогательный basis для maxvol-like residual updates. |

### Routing Методы

| Метод | Назначение |
|---|---|
| `predict_partitions(X)` | Возвращает argmax partition для новых объектов. |
| `predict_partition_proba(X)` | Возвращает row-wise probability по partitions. |
| `transform_embedding(X)` | Строит spectral embedding новых объектов. |
| `_project_new_unfolding(M_new)` | Делегирует backend-у projection новых rows в старый spectral basis. |
| `_routing_probability(embedding, active_centroids)` | Делегирует backend-у softmax по расстояниям до centroids. |

Routing probability:

```text
p(c | x) = softmax_c(-||z(x) - mu_c||_2^2 / T)
```

где `z(x)` - spectral embedding нового объекта, `mu_c` - centroid partition-а, `T` - `routing_temperature`. Затем применяется shrinkage:

```text
p_final = (1 - shrinkage) * p + shrinkage * uniform
```

### Text2Image Prompt: RMT Adaptive Rank

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Adaptive rank selection figure: singular value spectrum curve, cumulative explained variance curve, initial rank at 25 percent of matrix dimension, selected rank where 95 percent explained variance threshold is crossed. Include equations for EV(k).
```

### Text2Image Prompt: Leverage Chunk Selection

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Leverage score chunk selection diagram: spectral embedding points grouped into clusters, row leverage intensity indicated by subtle accent color, selected points inside each cluster, comparison of all, leverage, capped leverage PPS, maxvol, and hybrid selection methods; show clipping of extreme leverage scores at a quantile threshold.
```

## SpectralClusterSelector

Файлы:

- `sampling_zoo/core/sampling_strategies/spectral/cluster_selection.py`;
- `sampling_zoo/core/sampling_strategies/spectral/cluster_selection_contracts.py`;
- `sampling_zoo/core/sampling_strategies/spectral/cluster_consensus.py`.

Назначение: отдельный collaborator для кластеризации spectral embedding. Он разгружает `RMTContractionTensorSampler`: sampler строит embedding и partitions, а selector отвечает за candidates, scoring, hard constraints и выбор лучшего разбиения.

### Основные Методы

| Метод | Назначение |
|---|---|
| `select(embedding, target=None)` | Публичный метод: строит candidates, выбирает лучший и возвращает `ClusterSelectionResult`. |
| `build_candidate_plan(n_samples)` | Чисто строит typed plan: исходную сетку `k`, size-guard rejections, fallback и adapter requests. |
| `_build_candidates(...)` | Исполняет `ClusterCandidatePlan` с tqdm `Spectral cluster candidates` и собирает successes/failures. |
| `_fit_candidate_request(...)` | Превращает исключение optional adapter-а в `ClusterCandidateFitFailure`, не скрывая причину. |
| `_fit_count_based_candidate(...)` | Обучает `kmeans`, `bisecting_kmeans` или `gmm` для заданного `k`. |
| `_fit_hdbscan_candidate(...)` | Строит HDBSCAN candidate через sklearn/external backend или возвращает typed unavailable failure. |
| `_score_components(...)` | Определяет effective target type и делегирует pure core расчёт silhouette, balance, regression target contrast или classification partition profile. |
| `_classification_components(...)` | Строит global и per-cluster class counts и передает их в pure `evaluate_classification_partition_components(...)`. |
| `_build_scoring_context(...)` | Один раз определяет target type и, для opt-in режима, создаёт общий `PartitionValidationProxyEvaluator` для всех candidates. |
| `_candidate_score(...)` | Для `balanced_silhouette` применяет balance, classification и hard-constraint слагаемые к одному immutable `ClusterScoreComponents`. |
| `_select_by_weighted_vote(...)` | Агрегирует candidates по числу clusters через soft weights от score. |
| `_select_by_coassociation(...)` | Строит sparse weighted membership representation, получает consensus labels и проверяет их прежним balanced objective. |
| `build_weighted_membership_embedding(...)` | Чисто строит разреженную матрицу memberships без материализации квадратной co-association matrix. |

### Typed Contracts

- `ClusterCandidatePlan` различает исходную и допустимую count grid, хранит
  `size_guard_rejections` и явный `size_guard_fallback_applied`;
- `ClusterCandidateRequest` описывает один count-based или density-based adapter call;
- `ClusterCandidateFitFailure` различает `adapter_unavailable` и `fit_failed`;
- `ClusterScoreComponents` хранит objective inputs и список
  `ClusterConstraintViolation` вместо неявного boolean-only результата;
- `ClassificationPartitionComponents` хранит class counts, missing-class fraction,
  долю одно-классовых chunks/строк и class-distribution drift;
- `PartitionValidationPlan` хранит единый deterministic train/validation split,
  одинаковый для всех candidate algorithms и значений `k`;
- `PartitionValidationComponents` хранит baseline/candidate loss, relative gain,
  routed validation counts и долю fallback rows;
- `ClusterConsensusPlan` хранит нормированные source weights, weighted votes по `k`,
  выбранное число clusters и размер sparse representation;
- `ClusterSelectionUnavailableError` содержит весь plan и failures, если не удалось
  построить ни одного candidate.

Старые поля `ClusterSelectionResult.candidates` и словарные diagnostics сохранены.
Typed contracts являются внутренним source of truth, а словари материализуются на
границе sampler/reporting для обратной совместимости.

### Co-association Consensus

Для каждого source partition `m` строится one-hot membership matrix `Z_m`, а score
превращается в softmax weight `w_m`. Разреженное представление имеет вид

\[
H = [\sqrt{w_1}Z_1\;|\;\sqrt{w_2}Z_2\;|\;\cdots\;|\;\sqrt{w_M}Z_M].
\]

Его Gram matrix точно равна weighted co-association matrix:

\[
HH^\top = \sum_{m=1}^{M} w_m Z_m Z_m^\top,
\]

где элемент `(i,j)` показывает взвешенную долю candidates, поместивших строки `i`
и `j` в один cluster. Реализация не создаёт `n_samples x n_samples` matrix: KMeans
работает прямо на sparse `H`, где число non-zero элементов не превышает
`n_samples * n_candidates`.

Целевое `k` выбирается совместимым weighted vote. В отличие от legacy policy,
финальные labels принадлежат consensus partition, а не одному source algorithm.
Consensus повторно проходит balanced objective; при нарушении hard constraints
selector возвращает лучший допустимый source candidate того же `k` и фиксирует
`fallback_to_source_candidate`. Поле `score_delta_vs_best_source` отдельно показывает
разницу objective, но не превращает policy в неявный best-of-two выбор.

### Balanced Silhouette

`balanced_silhouette` нужен потому, что чистый silhouette часто выбирает слишком малое число clusters и не штрафует практические проблемы chunk training. В score добавлены:

- penalty за `max_cluster_imbalance_ratio`;
- penalty за tiny clusters ниже `min_cluster_fraction`;
- optional `target_contrast`, если хочется поощрять target-различимость clusters;
- для classification: penalty за отсутствующие классы, одно-классовые chunks и drift распределения классов;
- hard constraint penalty, если cluster candidate нарушает ограничения.

Для classification objective имеет вид

\[
S = s_{sil}
- \lambda_{imb}p_{imb}
- \lambda_{tiny}p_{tiny}
- \lambda_{miss}p_{miss}
- \lambda_{single}p_{single}
- \lambda_{drift}p_{drift}
- p_{hard}.
\]

Здесь `p_miss` — доля нулевых элементов в матрице chunk/class counts,
`p_single` — доля chunks только с одним представленным классом, а `p_drift` —
среднее total-variation distance локальных class distributions от глобального,
взвешенное по размеру chunk. Одно-классовый chunk добавляет
`single_class_cluster` в hard violations. Отсутствие отдельных редких классов
остается мягким штрафом: требование иметь каждый класс в каждом chunk часто
невыполнимо на multiclass datasets.

Classification-слагаемые отсутствуют для regression, поэтому прежняя regression
формула и candidate scores сохраняются. В benchmark-конфигах тип задачи задается
явно; при прямом использовании sampler-а с целочисленной regression target следует
также передавать `cluster_target_type="regression"`.

### Validation Proxy

`cluster_selection_metric="validation_proxy"` является явной ablation policy и
не заменяет default `balanced_silhouette`. Selector создаёт один внутренний holdout
внутри outer train fold и использует его для всех candidates. Для каждого candidate:

1. centroids считаются только по proxy-train строкам;
2. proxy-validation строки hard-route-ятся к ближайшему centroid;
3. regression local expert предсказывает среднее target своего partition;
4. classification local expert предсказывает сглаженное распределение классов;
5. слишком маленький local expert заменяется global baseline.

Для regression используется RMSE, для classification — log loss. Итоговый gain:

\[
G = \frac{L_{global} - L_{partition}}{\max(|L_{global}|, \varepsilon)}.
\]

Candidate score равен `G - hard_constraint_penalty`; valid pool по-прежнему
учитывает imbalance, tiny clusters и одно-классовые chunks. Положительный `G`
означает, что простые routed local experts лучше global constant baseline.

Это proxy качества partition/router, а не вложенное обучение реальных TabPFN или
LightGBM experts. Clustering candidates строятся на всём outer train fold, но test
fold и внешняя validation выборка benchmark-а в selection не участвуют. Для
подтверждения эффекта policy нужно сравнивать downstream benchmark metrics.

### Text2Image Prompt: Cluster Selection

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Spectral cluster consensus figure: a typed candidate plan branches into KMeans, bisecting KMeans, Gaussian mixture, and HDBSCAN; adapter failures remain structured; candidate scores become normalized weights; weighted one-hot membership blocks form a sparse matrix H; annotate H H transpose equals the weighted co-association matrix without materializing it; sparse KMeans produces consensus labels and hard-constraint validation can fall back to the best source partition. Show the pure-core and effect-shell boundary, labeled dimensions n by sum k, and a small valid/fallback decision panel.
```

### Text2Image Prompt: Class-Aware Partition Objective

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Classification-aware spectral partition scoring figure with three candidate chunkings of the same binary dataset: balanced class coverage, missing-class chunks, and single-class chunks. Show a chunk-by-class count matrix, global class distribution, weighted total-variation drift, soft penalties for missing classes and drift, and a red hard-constraint marker for single-class chunks. End with a balanced-silhouette score equation and selected candidate panel.
```

### Text2Image Prompt: Validation-Driven Partition Proxy

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Validation-driven spectral partition selection figure: one outer training fold creates a shared internal train and validation split; several clustering candidates with different k values use train-only centroids; validation rows route to local constant experts; regression RMSE or classification log loss is compared with a global baseline; relative validation gain and hard constraints select the candidate. Include a fallback arrow from undersized local experts to the global baseline and a clear note that the test fold is never used.
```

## MatrixRMTBackend И TensorRMTBackend

Файлы:

- `sampling_zoo/core/sampling_strategies/spectral/backend/matrix_backend.py`
- `sampling_zoo/core/sampling_strategies/spectral/backend/tensor_backend.py`

Назначение: backend-слой для RMT linear algebra. Sampler не должен напрямую импортировать torch или sklearn randomized SVD для backend-примитивов.

### Общие Backend Примитивы

| Метод | Matrix backend | Tensor backend |
|---|---|---|
| `build_mode0_unfolding(X, view_specs)` | NumPy matrix operations | Torch tensor operations |
| `compute_spectral_basis(M, rank)` | sklearn randomized SVD | torch randomized SVD |
| `project_new_unfolding(M_new, basis)` | NumPy projection | Torch projection |
| `routing_probability(embedding, centroids, temperature, shrinkage)` | NumPy softmax distances | Torch softmax distances |

### RMTSpectralBasis

`RMTSpectralBasis` хранит:

- `U`: left singular vectors;
- `singular_values`;
- `Vt`: right singular vectors;
- `leverage_scores`;
- `truncate(rank)`: возвращает basis с меньшим rank.

### Text2Image Prompt: Backend Split

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Backend separation diagram: RMT sampler orchestration at top, MatrixRMTBackend on left with NumPy and sklearn kernels, TensorRMTBackend on right with torch CUDA kernels. Shared primitive interface in the center: unfolding, SVD, projection, routing.
```

## RMTPartitionSelectionAblationOrchestrator

Файл: `examples/benchmark/rmt_partition_selection_ablation.py`

Назначение: запускает изолированное downstream-сравнение
`balanced_silhouette` и `validation_proxy`, не умножая основной medium grid на
новую ось.

### Контракты и методы

| Элемент | Назначение |
|---|---|
| `RMTPartitionSelectionAblationConfig` | Frozen scientific config. Нормализует datasets, models, budgets и selection metrics; фиксирует `routed_weighted`, spectral router и gaussian views через преобразование в основной regression config. Параметр `model_n_jobs` явно ограничивает число потоков каждой parallel estimator; серверный default равен `1`, чтобы исключить oversubscription на многоядерных узлах. |
| `RMTPartitionSelectionGridPoint` | Typed leaf point из пары `(budget_ratio, cluster_selection_metric)` с уникальным config name. |
| `make_rmt_partition_selection_grid(...)` | Чисто и детерминированно строит budget-paired grid. |
| `make_rmt_partition_selection_strategy_configs(config)` | Материализует typed grid в legacy strategy kwargs только на границе benchmark runner. |
| `_build_experiment_plan()` | Включает все scientific axes и proxy parameters в run identity и artifact manifest. |
| `_build_strategy_grid()` | Нормализует materialized configs в `StrategyGridContract`. |
| `_make_model_pool()` | Материализует модели с явным `model_n_jobs` из ablation config; лимит одинаково применяется к LightGBM и RandomForest. |
| `_build_run_meta(...)` | Добавляет тип эксперимента и validation-proxy parameters в incremental metadata. |
| `run_rmt_partition_selection_ablation(...)` | Публичный entrypoint; execution lifecycle наследуется от `RMTRegressionExperimentOrchestrator`. |

Config names содержат `selection_balanced_silhouette` или
`selection_validation_proxy`. Это часть resume identity: результаты разных
критериев не считаются одним leaf run.

### Text2Image Prompt: Partition Selection Ablation

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Paired RMT partition-selection ablation figure: one shared dataset and spectral embedding branch into balanced silhouette and validation proxy selectors, both use identical Gaussian views, spectral router, routed weighted ensemble, model, seed and budget; downstream RMSE and timing flow into a paired delta table, with a clear label delta equals validation proxy minus balanced silhouette.
```

## RMTReportTableBuilder

Файл: `examples/benchmark/rmt_report_tables.py`

Назначение: превращает run records в аналитические таблицы для отчета.

### Основные Методы

| Метод | Назначение |
|---|---|
| `build_report_tables(run_records, output_dir, reference_metrics=None)` | Публичный builder method. |
| `_normalize_records(...)` | Приводит records к DataFrame. |
| `_build_raw_runs_table(df)` | Формирует raw table с dataset, sampler, model, budget, metrics, timings. |
| `_attach_rmse_baseline(raw)` | Добавляет baseline RMSE. |
| `_attach_rmse_drop(raw)` | Считает относительное ухудшение/улучшение относительно baseline. |
| `_build_efficiency_table(raw)` | Строит sample efficiency curve. |
| `_build_partition_selection_comparison(efficiency)` | Строит paired comparison `validation_proxy - balanced_silhouette` по одинаковым dataset/model/router/view/budget axes. |
| `_build_minimal_budget_table(efficiency)` | Находит минимальный budget для thresholds delta. |
| `_write_table(table, path)` | Записывает CSV. |

Raw RMT table дополнительно вытаскивает RMT-specific поля: `view_strategy`, `n_views`, `n_views_policy`, adaptive rank diagnostics, partition selection diagnostics, validation-proxy loss/gain/fallback columns и routing refinement columns (`routing_refinement_status`, `routing_refinement_stop_reason`, `routing_refinement_best_iteration`, `routing_refinement_metric_improvement`, `routing_refinement_final_imbalance`). Builder также пишет `partition_selection_comparison.csv`; отрицательная `rmse_delta_validation_proxy_minus_balanced_silhouette` означает, что validation proxy лучше на данной paired-конфигурации.

### Text2Image Prompt: Report Builder

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Report table builder diagram: raw run records enter transformation pipeline, baseline RMSE attachment, RMSE drop computation, sample efficiency curve, paired partition-selection comparison, minimal effective budget table, CSV artifacts.
```

## BenchmarkLogger

Файл: `examples/benchmark/benchmark_logging.py`

Назначение: пишет structured log artifacts на каждый strategy run. Он отделен от incremental saver: logger фиксирует benchmark event, saver отвечает за durable records и snapshot rebuild.

Типичные artifacts:

- `logs/strategy_runs.jsonl`;
- per-strategy JSON snapshots в `metrics/`;
- markdown/status logs, если включены в runner.

### Text2Image Prompt: Logger

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Structured benchmark logging figure: fold execution emits metrics, timings, sample stats, extra diagnostics; logger writes event JSONL and metric snapshots; incremental saver consumes the same record for durable experiment-level state.
```

## Benchmark Model Registry

Файл: `examples/benchmark/benchmark_models.py`

Назначение: строит model pool для benchmark-а и изолирует optional dependencies.

### Важные Helpers

| Helper | Назначение |
|---|---|
| `_load_torch_modules()` | Лениво импортирует torch/nn/optim и кеширует результат через `_TORCH_IMPORT_ATTEMPTED`. |
| `_load_tabpfn_classes()` | Лениво импортирует `TabPFNClassifier`, `TabPFNRegressor`, `ModelVersion`; отсутствие пакета не ломает import module. |
| `_load_tabicl_classes()` | Лениво импортирует TabICL classes. |
| `_resolve_tabpfn_device()` | Выбирает `cuda`, если активный torch backend видит CUDA, иначе `cpu`, с env override `TABPFN_DEVICE`. |
| `_make_tabpfn_kwargs(...)` | Передает TabPFN `device`, `random_state`, `n_estimators`; на CPU включает `ignore_pretraining_limits` только при явном env override. |

Правило: optional heavy packages нельзя импортировать на верхнем уровне benchmark module. Это особенно важно для окружений, где torch CUDA версия подбиралась вручную под конкретную GPU.
