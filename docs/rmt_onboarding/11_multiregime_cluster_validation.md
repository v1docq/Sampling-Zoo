# Multi-regime cluster validation для RMT

## Цель

OpenML benchmark показывает итоговую ошибку ансамбля, но не отвечает, почему было
выбрано неверное число partitions. Spiked benchmark из
`10_spiked_synthetic_validation.md` отдельно проверяет восстановление одного
глобального low-rank подпространства. Multi-regime benchmark продолжает эту цепочку и
изолирует следующий вопрос:

> восстанавливает ли RMT sampler известное число локальных режимов и их принадлежность,
> если spectral embedding уже содержит достаточно сигнала?

Benchmark диагностический. Он не меняет production defaults семплера и не использует
истинные labels при обучении auto-policy.

## Архитектурная граница

Чистый модуль `sampling_zoo/core/validation/multi_regime_models.py` содержит:

- `MultiRegimeDataConfig` - immutable и валидированный контракт генератора;
- `MultiRegimeDataset` - наблюдаемую матрицу, signal/noise decomposition, истинные
  regime labels и signal subspace;
- `MultiRegimeValidationGridPoint` - один детерминированный leaf-run;
- `ClusterRecoveryMetrics` - label-permutation-invariant результат;
- pure functions построения grid, генерации данных и оценки clustering.

Effectful shell находится в
`examples/benchmark/run_rmt_multiregime_synthetic_experiment.py`. Он загружает
backend, запускает sampler, показывает tqdm progress и инкрементально записывает
артефакты. Общая для synthetic benchmark-ов логика run identity, manifest, JSONL и
выбора Torch device вынесена в `synthetic_benchmark_runtime.py`.

## Математическая модель

Для режима (c\in\{1,\ldots,K\}) строки сигнала строятся как

\[
L_i = a\mu_c + b z_i A_c^\top,
\qquad z_i\sim\mathcal N(0,I_r),
\]

где (mu_c) задает separation между режимами, (A_c) задает локальное
подпространство ранга (r), а направления разных режимов ортогональны. После
глобального центрирования наблюдается

\[
X=L+N,
\qquad
\operatorname{SNR}_F=
\frac{\lVert L\rVert_F^2}{\lVert N\rVert_F^2}.
\]

Шум Gaussian или Student-t центрируется и масштабируется до точного заданного SNR
для каждой реализации. При невырожденных факторах истинный ранг сигнала равен

\[
r_{signal}=K r + (K-1),
\]

где (Kr) - локальные направления, а (K-1) - пространство центрированных
межрежимных сдвигов.

Поддерживаются два профиля размера режимов:

- `balanced` - почти равные размеры;
- `imbalanced_4_to_1` - геометрически убывающие веса с отношением крупнейшего
  режима к наименьшему около 4:1.

## Probe policies

Один и тот же dataset и spectral configuration сравниваются в трех режимах:

| Policy | Назначение | Candidate policy |
|---|---|---|
| `fixed_oracle` | Control верхнего уровня: можно ли восстановить labels при известном истинном (K) | fixed KMeans с `n_partitions=K` |
| `auto_production` | Текущая production-like логика | `min_auto_partition_size=256` |
| `auto_unrestricted` | Ablation size guard | та же objective и adapters, но `min_auto_partition_size=1` |

`fixed_oracle` не является production-предложением. Он разделяет ошибки embedding и
ошибки выбора числа clusters.

Важная деталь текущего selector-а: size guard применяется к count-based алгоритмам
`kmeans`, `bisecting_kmeans` и `gmm`, но HDBSCAN сам определяет число clusters. Поэтому
отчет хранит отдельно:

- `candidate_set_contains_true_n` - истинное K присутствует среди всех реально
  построенных candidates;
- `planned_count_candidate_set_contains_true_n` - истинное K присутствует в typed
  count plan после size guard/fallback, до запуска adapters;
- `count_based_candidate_set_contains_true_n` - count-based adapter фактически
  построил candidate с истинным K;
- `size_guard_rejected_true_n` и `size_guard_fallback_applied` отделяют первичный
  rejection от возврата всей grid при полном отсеве;
- `density_candidate_rescued_true_n` - count grid отсек истинное K, но HDBSCAN вернул
  candidate с правильным числом clusters;
- `candidate_failure_count` и `candidate_failure_codes` показывают недоступные или
  упавшие adapters, которые раньше молча исчезали из результата.

Если size guard отсекает все count-based значения, текущий selector возвращает
исходную сетку как fallback. Этот behavior покрыт invariant test и должен учитываться
при интерпретации малых smoke datasets.

## Метрики восстановления

Число clusters оценивается полями `predicted_n_clusters`, signed/absolute count error и
`cluster_count_exact`.

Качество labels не зависит от произвольной нумерации кластеров:

- `adjusted_rand_index` и `normalized_mutual_information` измеряют согласованность
  разбиений;
- `aligned_accuracy` использует Hungarian matching между истинными и найденными
  labels;
- `purity` измеряет доминирующий истинный режим внутри каждого найденного cluster;
- `mean_true_regime_recall` и `min_true_regime_recall` показывают, не потерян ли
  отдельный режим;
- `predicted_imbalance_ratio` и `predicted_min_cluster_fraction` контролируют
  геометрию partitions.

Параллельно `selected_subspace_recall` сравнивает истинный signal span и выбранный
sampler basis. Высокий subspace recall при низком ARI указывает на cluster-selection
слой, а не на spectral engine.

## Диагностическое дерево

1. Низкий `fixed_oracle` ARI при высоком SNR означает проблему embedding, rank policy
   или недостаточную separability генератора.
2. Высокий oracle ARI и низкий count-based coverage означают, что истинный K был
   исключен planner guard до расчета balanced objective.
3. Истинный K присутствует, но auto-policy выбирает другой K: проблема objective,
   hard constraints, weighted vote или clustering adapter.
4. Count grid не содержит K, но `density_candidate_rescued_true_n=True`: результат
   спасен HDBSCAN; это надо отделять от корректности count planner-а.
5. NumPy/Torch paired delta выше tolerance означает backend divergence, а не качество
   selector-а.

## Запуск

Минимальная локальная проверка:

```powershell
python examples\benchmark\run_rmt_multiregime_synthetic_experiment.py --smoke
```

Стандартная сетка:

```powershell
python examples\benchmark\run_rmt_multiregime_synthetic_experiment.py
```

Расширенная server grid:

```powershell
python examples\benchmark\run_rmt_multiregime_synthetic_experiment.py --server-grid
```

Server grid содержит 11 520 leaf-runs: 16 SNR, три значения K, 10 seed, две noise
family, два balance profile, два backend и три probe policy. Перед ним необходимо
запустить smoke в том же Torch/CUDA окружении и оценить время на небольшой подвыборке.

Публичный Python entrypoint:

```python
from examples.benchmark.run_rmt_multiregime_synthetic_experiment import (
    run_rmt_multiregime_synthetic_experiment,
)

output_dir = run_rmt_multiregime_synthetic_experiment(
    snr_values=(0.1, 0.3, 1.0, 3.0, 10.0),
    n_regimes_values=(2, 3, 5),
    seeds=range(5),
    backends=("numpy", "torch"),
)
```

## Артефакты

Каждый leaf-run сначала fsync-записывается в
`metrics/rmt_multiregime_runs.jsonl`. Snapshot builder создает:

- `metrics/rmt_multiregime_raw_runs.csv`;
- `metrics/rmt_multiregime_summary_by_snr.csv`;
- `metrics/rmt_multiregime_algorithm_frequency.csv`;
- `metrics/rmt_multiregime_policy_regret.csv`;
- `metrics/rmt_multiregime_backend_agreement.csv`;
- `report.md`, `run_meta.json`, `artifact_manifest.json`.

Недоступный Torch backend записывается как `skipped`, ошибка отдельного leaf-run как
`failed`; grid продолжает работу, а предыдущие результаты не теряются.

## Критерий следующего изменения production policy

Изменять `min_auto_partition_size`, balanced objective или ensemble voting стоит только
после парного анализа:

1. `fixed_oracle` устойчиво восстанавливает regimes после detectability threshold;
2. subspace recall достаточно высок и согласован между backend;
3. auto-policy regret локализован либо в candidate coverage, либо в objective;
4. эффект сохраняется на Gaussian и Student-t noise и не объясняется одним seed;
5. предлагаемая policy улучшает recovery без взрывного роста tiny/imbalanced clusters.

## Text2Image Prompt

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Four-panel figure for multi-regime RMT cluster validation: panel A mixture of K affine low-rank regimes with orthogonal centroids and local subspaces, balanced and four-to-one imbalanced profiles, plus Gaussian or Student-t noise with exact Frobenius SNR; panel B random feature contractions and sv-scaled sample embedding; panel C three probe paths labeled fixed oracle K, auto production size guard, and auto unrestricted, branching into KMeans, bisecting KMeans, Gaussian mixture, and HDBSCAN; panel D diagnostic decision tree comparing subspace recall, count-based candidate coverage, HDBSCAN density rescue, adjusted Rand index, aligned accuracy, and auto-policy regret versus oracle. Include the equations X equals L plus N and signal rank equals K times local rank plus K minus one, thin confidence bands over seeds, precise scientific labels.
```
