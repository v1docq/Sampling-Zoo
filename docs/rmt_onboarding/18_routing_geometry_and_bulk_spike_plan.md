# RMT routing geometry и bulk/spike experts: актуальный план разработки

Дата актуализации: 2026-08-10. Статус: `phase_a2_cross_fitted_selector_implemented`.

Формальная постановка проверки гипотез: [Эксперимент: routing geometry и bulk/spike experts](../rmt_contraction_algo/RMT%20routing%20geometry%20и%20bulk-spike%20experiment.md).

Этот документ фиксирует следующий механизм исследования после classification gate: геометрию spectral routing и component-aware разделение `bulk/spike`.

![Различие спектральных spikes, leverage строк и иерархического ансамбля](../img/4.RMT_bulk_spike_moe_v2.png)

## Статус реализации

| Этап | Статус | Реализация |
|---|---|---|
| P0 | завершен | Текущий L2-router сохранен как arm A1; leverage cap и RMT bulk edge разведены терминологически. |
| P1 | завершен | `routing_contracts.py`, `PartitionGeometryBuilder`, совместимый `predict_partition_proba(...)`. |
| P2 | завершен для Phase A | Matrix/Torch kernels для scaled L2, shrinkage Mahalanobis, cosine и GMM posterior; row-wise routing diagnostics. |
| P3 | завершен | Offline replay фиксированных expert outputs, validation-only temperature calibration, incremental artifacts и synthetic smoke. |
| P3.1 | реализован, ожидает real-data gate | `ValidationRoutingGeometrySelector`, независимые calibration/selection holdouts и A7 selector arm с A2 fallback. |
| P3.2 | реализован, ожидает малую проверку на реальных данных | `CrossFittedRoutingGeometrySelector`, попарные оценки на внутренних фолдах, надежный возврат к A2 и селектор A8. |
| P4 | не начат | Component split и row participation реализуются только после real-data gate Phase A. |
| P5-P6 | заблокированы gate-ом | Bulk/spike experts и full grid не запускаются до выбора routing geometry. |

Целевые сценарии запуска: `examples/benchmark/rmt_routing_geometry_experiment.py` для первичного отбора, `examples/benchmark/rmt_validation_geometry_selector_experiment.py` для A7 и `examples/benchmark/rmt_cross_fitted_geometry_selector_experiment.py` для устойчивой проверки A8. Быстрая локальная проверка: `examples/benchmark/rmt_routing_geometry_smoke.py`.

### Результат Phase A и следующий gate

Phase A завершен без ошибок: `1120/1120` leaf runs, восемь датасетов, пять seeds, четыре бюджета и семь routing arms. Основные выводы:

- `median_scaled_euclidean` имеет лучший средний ранг `2.28` и является fallback;
- cosine arm имеет средний ранг `2.53`, чаще становится абсолютным победителем и больше не считается negative control;
- GMM posterior имеет средний ранг `3.21` и особенно полезен для classification;
- squared Euclidean дает переуверенный routing и остается только воспроизводимым baseline;
- diagonal/full Mahalanobis не проходят в основной selector до отдельной temperature/covariance ablation.

A7 показал, что выбор по одному разбиению валидационной выборки чувствителен к случайному составу строк. Поэтому A8 выбирает один вариант из A2/A5/A6 по пяти внутренним фолдам. На каждом фолде `temperature` и `expert_priors` оцениваются только по калибровочным строкам, а качество измеряется на непересекающихся строках. Внешняя тестовая выборка оценивается один раз после выбора. Синтетический селектор называется `A8_cross_fitted_selected`; имя `B0` остается зарезервированным для bulk/spike topology.

## 1. Принятые решения

1. Текущий `squared Euclidean + softmax` остается воспроизводимым baseline, но перестает быть неявно единственной routing geometry.
2. `median_scaled_euclidean`, shrinkage Mahalanobis и GMM posterior исследуются последовательно, а не одной большой сеткой.
3. Нормировка на медианное расстояние выполняется отдельно для каждой partition. Общая сумма медиан эквивалентна изменению глобальной температуры и не корректирует разные радиусы clusters.
4. `GMM posterior` считается отдельным probabilistic kernel, а не разновидностью расстояния.
5. Порог `Q_0.95` для leverage ограничивает влияние строк. Он не является RMT bulk edge.
6. Bulk/spike разделение строится по компонентам спектра через empirical null edge, а не по квантилю row leverage.
7. Иерархическая topology `bulk expert + spike experts` остается opt-in research mode до прохождения synthetic и real-data gates.

## 2. Текущий baseline и его ограничения

Сейчас sampler хранит один arithmetic centroid для каждой partition. Даже если partition labels получены через GMM, HDBSCAN или consensus clustering, inference routing сводит partition к среднему вектору.

Для `embedding_mode="sv_scaled"`:

\[
z_i=U_{i,:}\Sigma,
\qquad
c_k=\frac{1}{|C_k|}\sum_{i\in C_k}z_i.
\]

Backend вычисляет:

\[
d_k^2(x)=\|z(x)-c_k\|_2^2,
\qquad
r_k(x)=\operatorname{softmax}_k\left(-d_k^2(x)/T\right).
\]

После uniform shrinkage ensemble умножает base routing на validation reliability эксперта. Такой pipeline предполагает сферические clusters сопоставимого масштаба и совпадение геометрической близости с областью компетенции expert model.

## 3. Целевая архитектурная граница

Routing geometry должна быть отделена от ensemble-level aggregation:

```text
spectral embedding + partition labels
  -> PartitionGeometryBuilder
  -> PartitionGeometryContract
  -> backend distance / density kernel
  -> RoutingWeightContract
  -> RoutedWeightedRouter + validation priors
  -> expert aggregation
```

`RMTContractionTensorSampler` владеет spectral embedding и partition geometry. `RoutedWeightedRouter` продолжает владеть комбинацией base routing, validation priors и learned gates. `SamplingEnsemble` не получает numerical kernels.

### 3.1. Новые contracts

| Contract | Назначение |
|---|---|
| `PartitionGeometrySpec` | Валидированный выбор representation, metric, kernel, temperature и covariance policy. |
| `PartitionGeometryContract` | Names, centers/medoids, per-partition scales, priors, covariance or precision matrices, optional prototypes. |
| `RoutingDistanceContract` | Матрица distance/log-density, shape metadata и numerical diagnostics. |
| `RoutingWeightContract` | Нормированные row-wise weights и calibration metadata. |
| `SpectralComponentSplitContract` | Empirical bulk edge, spike/bulk masks, selection frequencies и reason. |
| `RowSpectralParticipationContract` | Leverage, spike energy, bulk energy, signalness и spike signatures. |
| `BulkSpikeTrainingPlan` | Exact budget allocation между bulk и spike experts без скрытого роста data budget. |

Все contracts должны быть immutable dataclasses. Raw config преобразуется в typed spec один раз, а backend получает только численные arrays и валидированные параметры.

### 3.2. Новые collaborators

| Модуль | Ответственность |
|---|---|
| `spectral/routing_geometry.py` | Pure построение centers, medoids, scales, covariance estimates и routing plans. |
| `spectral/routing_contracts.py` | Enums и immutable contracts. |
| `spectral/bulk_spike.py` | Component split, row participation и pure bulk/spike training plan. |
| `spectral/backend/matrix_backend.py` | NumPy kernels для distances, log-determinants и posterior normalization. |
| `spectral/backend/tensor_backend.py` | Torch/CUDA parity для тех же kernels. |
| `utils/ensemble_routing.py` | Только ensemble-level priors, learned gating и routing diagnostics. |

## 4. Конфигурационный интерфейс

Предлагаемый закрытый mode space:

```python
routing_representation = "source_centroid" | "sampled_centroid" | "medoid" | "gaussian" | "multi_prototype"
routing_metric = "squared_euclidean" | "median_scaled_euclidean" | "diag_shrinkage_mahalanobis" | "full_shrinkage_mahalanobis"
routing_kernel = "softmax" | "student_t" | "gmm_posterior"
routing_temperature = 1.0 | "auto"
routing_scale_scope = "global" | "per_partition"
routing_covariance_shrinkage = "auto" | float
routing_prototypes_per_partition = int
```

`gmm_posterior` требует representation с covariance и priors. Несовместимые combinations должны отклоняться typed validation error, а не исправляться скрытым fallback.

Bulk/spike research mode:

```python
spectral_component_policy = "explained_variance" | "empirical_null_edge"
expert_topology = "partition_local" | "bulk_spike_hierarchical"
bulk_spike_gate = "energy_ratio"
spike_expert_policy = "single" | "signature_clusters"
spike_budget_policy = "excess_energy" | "fixed_share"
```

Default остается `spectral_component_policy="explained_variance"` и `expert_topology="partition_local"` до завершения эксперимента.

## 5. Этапы реализации

### P0. Терминология и baseline snapshot

- заменить термин «спектральные выбросы строк» на «экстремальные leverage-оценки»;
- зафиксировать текущие formulas, defaults и backend parity test;
- сохранить current L2 routing как immutable comparison arm.

### P1. Routing geometry contracts

- добавить typed specs/contracts и deterministic normalization;
- извлечь построение partition representatives из cluster selector/sampler;
- сохранить public `predict_partition_proba(...)`.

### P2. Routing kernels и diagnostics

- реализовать median-scaled L2;
- реализовать diagonal и full shrinkage Mahalanobis;
- реализовать GMM log posterior с stable log-sum-exp;
- добавить validation-calibrated temperature;
- добавить oracle expert regret, top-1 margin, distance concentration, covariance condition number, dead-expert rate и centroid stability.

### P3. Offline routing replay

- сохранять validation/test embeddings и per-expert predictions/probabilities;
- пересчитывать routing variants без повторного fit sampler-а и experts;
- выбирать geometry только по inner validation и один раз оценивать test.

После P3 можно запускать первую новую real-data ablation.

### P3.1. Validation-selected routing geometry

- калибровать температуры A2/A5/A6 на calibration holdout;
- выбирать geometry на отдельном selection holdout через immutable policy/result contracts;
- предпочитать A2 при ничьей или недостаточном улучшении;
- сохранять candidate scores, причину выбора, fallback flag и выбранный arm;
- не передавать outer-test данные в selector;
- запускать A7 gate до начала bulk/spike topology.

### P3.2. Устойчивый выбор геометрии по внутренним фолдам

- использовать всю валидационную выборку как набор внутренних фолдов, не отделяя единственную выборку для выбора геометрии;
- для каждого фолда оценивать `temperature` и веса надежности экспертов только на его калибровочной части;
- вычислять попарный относительный выигрыш A5 и A6 относительно A2 на отложенной части каждого фолда;
- пропускать кандидата только при неотрицательных среднем и медианном выигрыше, положительном выигрыше как минимум на двух третях фолдов и нижней границе 95% bootstrap-интервала не ниже `-0.005`;
- для regression дополнительно применять те же ограничения по `MAE` как устойчивой метрике;
- возвращать явный статус `fallback_to_a2`, если ни один кандидат не прошел все ограничения;
- сохранять оценки каждого фолда, сводку обоснований и итоговое решение до обращения к внешней тестовой выборке.

Первый запуск после слияния ограничен задачами `Brazilian_houses`, `elevators`, `adult`, `jannis`, пятью seeds и бюджетами `(0.01, 0.05, 0.10, 0.20)`. Ожидаемый объем: `4 datasets x 5 seeds x 4 budgets x 4 arms = 320` завершенных записей.

### P4. Component-aware spectral diagnostics

- включить `SpectralNullDiagnostic` в targeted runs;
- материализовать spike mask по empirical edge и stability frequency;
- сохранять basis, достаточный для component-wise row energy;
- вычислять `spike_energy`, `bulk_energy`, `signalness`, `spike_signature`;
- не менять rank selection default на основании одной diagnostic run.

### P5. Bulk/spike training topology

- реализовать exact-budget `BulkSpikeTrainingPlan`;
- добавить один broad bulk expert;
- добавить single-spike expert как прямую проверку исходной гипотезы;
- добавить несколько experts по spike-signature clusters;
- реализовать soft signal/bulk gate и Mahalanobis/GMM routing внутри signal branch.

### P6. Full-grid gate

Полная AMLB grid запускается только если лучший routing geometry улучшает downstream metric или oracle regret без неприемлемого worst-case degradation, а hierarchical topology проходит synthetic identifiability tests.

## 6. Обязательные invariants и tests

- routing weights имеют shape `(n_rows, n_active_partitions)`, конечны, неотрицательны и суммируются в 1;
- identity covariance Mahalanobis совпадает с squared L2;
- median-scaled L2 инвариантен к общему изменению масштаба embedding;
- covariance shrinkage всегда дает положительно определенную матрицу;
- matrix и tensor backends согласованы в tolerance;
- GMM posterior учитывает priors и log-determinant;
- permutation partition names не меняет aligned predictions;
- empirical edge разделяет components, а не rows;
- total unique sampled rows и total trained rows явно фиксируются;
- outer test не участвует в temperature, geometry или topology selection.

## 7. Не входит в ближайший шаг

- перевод `bulk_spike_hierarchical` в default;
- одновременная перестройка clustering, routing geometry, rank policy и expert model;
- интерпретация softmax distance как calibrated posterior;
- отдельная упрощенная capacity для bulk expert до проверки topology при одинаковых model families.

## 8. Теоретические источники

- Baik, Ben Arous, Peche: [phase transition for spiked covariance matrices](https://arxiv.org/abs/math/0403022).
- Benaych-Georges, Nadakuditi: [low-rank perturbations of rectangular random matrices](https://arxiv.org/abs/1103.2221).
- Drineas et al.: [statistical leverage as row norms of left singular vectors](https://www.jmlr.org/papers/v13/drineas12a.html).
- Gavish, Donoho: [optimal singular-value shrinkage](https://arxiv.org/abs/1405.7511).
