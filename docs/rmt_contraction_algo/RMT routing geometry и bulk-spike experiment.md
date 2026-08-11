# Эксперимент: routing geometry и bulk/spike experts

Дата постановки: 2026-08-05. Актуализация: 2026-08-06. Статус: `phase_a_completed_selector_gate_ready`.

Связанный архитектурный план: [RMT routing geometry и bulk/spike experts](../rmt_onboarding/18_routing_geometry_and_bulk_spike_plan.md).

![Схема разделения bulk и spikes и иерархического ансамбля](../img/4.RMT_bulk_spike_moe_v2.png)

## 1. Исследовательские вопросы

Эксперимент должен ответить на два разных вопроса.

1. Улучшает ли downstream-качество замена текущего `squared Euclidean + softmax` на геометрию, учитывающую масштаб и анизотропию partitions?
2. Дает ли RMT-разделение спектральных компонент на `bulk` и устойчивые `spikes` преимущество перед row-level `capped_leverage` при одинаковом уникальном data budget?

Эти вопросы проверяются последовательно. Phase B не начинается до выбора рабочей routing geometry в Phase A.

Реализованный entrypoint Phase A:

```python
from examples.benchmark.rmt_routing_geometry_experiment import (
    run_rmt_routing_geometry_experiment,
)

results = run_rmt_routing_geometry_experiment(
    models=("lightgbm",),
    budget_ratios=(0.01, 0.05, 0.10, 0.20),
    seeds=(42, 43, 44, 45, 46),
)
```

Phase A завершен: `1120/1120` records, `0 failed`. В основной selector прошли A2 median-scaled Euclidean, A5 GMM posterior и A6 cosine. A2 и A6 статистически не различаются в попарном сравнении, а A5 дополняет их на classification datasets, поэтому один глобальный arm не фиксируется без дополнительного holdout gate.

Каждый набор expert-моделей обучается один раз для `dataset x seed x budget x model`. Arms A0-A6 используют одинаковые expert outputs. Температура выбирается по validation, после чего test оценивается один раз с зафиксированным значением. Результаты инкрементально сохраняются в `routing_geometry_runs.jsonl`, `routing_geometry_replay.csv` и `run_meta.json`.

## 2. Общие правила

- Единица сравнения: один `dataset x split x seed x budget_ratio x model`.
- Бюджеты: `(0.01, 0.05, 0.10, 0.20)` от training split.
- Число повторов: пять `seed` для inner validation, sampler, clustering и expert models при общем фиксированном OpenML outer test split.
- Основная модель screening-этапа: LightGBM. TabPFN запускается только для прошедших gate вариантов.
- Внешний test split используется один раз после выбора конфигурации на inner validation.
- Для каждой пары сравнений фиксируются preprocessing, fold, sampling seed и model seed.
- `total_unique_train_rows` должен совпадать между сравниваемыми arms. Перекрытия выборок и сумма строк, увиденных всеми experts, записываются отдельно.
- Ошибка одного leaf run сохраняется как структурированный результат и не прерывает остальные запуски.

## 3. Наборы данных

### 3.1. Synthetic gate

Перед реальными данными используются контролируемые генераторы:

1. Слабый сигнал ниже empirical bulk edge.
2. Один устойчивый spike выше edge.
3. Несколько ортогональных spikes разной силы.
4. Анизотропные clusters с различными covariance matrices.
5. High-leverage row anomaly, не связанная с информативным spike.
6. Classification data с редким классом и ограничением на class coverage чанков.

Synthetic gate считается пройденным, если алгоритм не создает spike experts ниже edge, обнаруживает устойчивый сигнал выше edge и Mahalanobis/GMM routing превосходит L2 на анизотропном сценарии.

### 3.2. Real-data screening

Минимальный сбалансированный набор:

| Тип | Датасеты |
|---|---|
| Regression | `diamonds`, `elevators`, `Brazilian_houses`, `OnlineNewsPopularity` |
| Classification | `adult`, `bank-marketing`, `covertype`, `jannis` |

Если задача недоступна в актуальной OpenML suite, runner фиксирует `dataset_unavailable`, а замена выбирается из той же размерной и task-type группы до начала запуска.

## 4. Phase A: Routing Geometry Replay

### 4.1. Цель

Изолировать влияние определения partition geometry, расстояния и преобразования расстояний в веса. Обучение sampler и expert-моделей выполняется один раз. Затем validation/test embeddings и прогнозы experts кэшируются, а routing arms переигрываются без повторного обучения моделей.

Фиксируются:

- preprocessing и spectral basis;
- `view_strategy="gaussian"` и adaptive spectrum-stability `n_views`;
- `embedding_mode="sv_scaled"`;
- выбранный rank, partition labels и sampled row ids;
- expert models и их class-aligned probabilities либо regression predictions.

### 4.2. Arms

| ID | Representation | Metric / kernel | Назначение |
|---|---|---|---|
| A0 | none | uniform voting | Контроль без геометрического роутинга. |
| A1 | source centroid | squared Euclidean, `T=1` | Текущий воспроизводимый baseline. |
| A2 | source centroid | median-scaled Euclidean, calibrated `T` | Учет разного радиуса partitions. |
| A3 | source centroid | diagonal shrinkage Mahalanobis, calibrated `T` | Учет покоординатной анизотропии. |
| A4 | source centroid | full shrinkage Mahalanobis, calibrated `T` | Учет полной covariance geometry. |
| A5 | Gaussian component | regularized GMM posterior, optional tempering | Вероятностный routing с prior и объемом clusters. |
| A6 | source centroid | cosine similarity | Диагностический negative control. |

Температура выбирается только на inner validation. Для двух лучших metric/kernel вариантов дополнительно сравниваются `source_centroid`, `sampled_centroid` и `medoid`. Multi-prototype representation остается последующей абляцией и не входит в первую сетку.

### 4.3. Метрики

Downstream quality:

- regression: primary `RMSE`, secondary `MAE`;
- binary classification: primary `ROC AUC`, secondary `log_loss`, Brier score, ECE и macro F1;
- multiclass classification: primary `log_loss`, secondary Brier score, ECE, accuracy и macro F1.

Routing diagnostics:

- oracle expert regret: разница между loss смеси и loss лучшего expert для каждой строки;
- top-1 routing margin и mean max routing probability;
- routing entropy и effective expert count;
- Spearman correlation между routing weight и отрицательным per-expert loss;
- доля dead experts и фактическое распределение строк по experts;
- assignment imbalance;
- устойчивость centers/covariances между split/seed;
- condition number и доля regularization для covariance estimates.

### 4.4. Gate выбора geometry

Вариант проходит Phase A, если одновременно:

1. дает положительный median paired gain по primary metric относительно A1;
2. уменьшает oracle expert regret либо улучшает калибровку без ухудшения primary metric;
3. его 95% paired bootstrap CI не показывает систематического вреда;
4. worst-case degradation не превышает `0.005` AUC, `1%` log loss или `1%` RMSE относительно A1;
5. не вызывает collapse на одного expert или рост dead expert rate.

### 4.5. Phase A.1: validation-selected geometry

Цель этапа — проверить, переносится ли observed validation ranking A2/A5/A6 на outer test без повторного fit sampler и expert models.

```python
from examples.benchmark.rmt_validation_geometry_selector_experiment import (
    run_rmt_validation_geometry_selector_experiment,
)

results = run_rmt_validation_geometry_selector_experiment(
    models=("lightgbm",),
    budget_ratios=(0.01, 0.05, 0.10, 0.20),
    seeds=(42, 43, 44, 45, 46),
    selector_fraction=0.5,
)
```

Разбиение выполняется последовательно:

1. outer train/test остается фиксированным;
2. из outer train выделяется `20%` validation pool;
3. половина pool используется для temperature calibration и expert priors;
4. вторая половина выбирает geometry;
5. outer test используется один раз для A2, A5, A6 и A7 selected policy.

| ID | Политика | Назначение |
|---|---|---|
| A2 | fixed median-scaled Euclidean | безопасный fallback; |
| A5 | fixed GMM posterior | classification-oriented кандидат; |
| A6 | fixed cosine | directional geometry кандидат; |
| A7 | validation-selected A2/A5/A6 | проверка адаптивного выбора без test leakage. |

A7 проходит gate, если улучшает средний paired rank относительно fixed A2, не увеличивает worst-case degradation и выбирает test-победителя чаще любой фиксированной geometry. `B0` не используется на этом этапе: идентификатор зарезервирован для bulk/spike baseline.

### 4.6. Phase A.2: выбор геометрии по внутренним фолдам

Единственное разбиение для калибровки и выбора из Phase A.1 дает мало наблюдений и может выбрать вариант из-за случайного состава валидационных строк. Phase A.2 использует `K=5` внутренних фолдов. Для каждого варианта и фолда:

1. на калибровочной части выбирается `temperature`;
2. на той же части заново оцениваются веса надежности экспертов;
3. на отложенной части вычисляются основная метрика и, для regression, `MAE`;
4. внешняя тестовая выборка остается недоступной до окончательного выбора варианта.

Для кандидата `a` и базовой геометрии A2 относительный выигрыш на фолде `f` равен

\[
g_{a,f}=
\begin{cases}
\dfrac{L_{A2,f}-L_{a,f}}{|L_{A2,f}|+\varepsilon},
& \text{для метрик, где меньше лучше},\\[6pt]
\dfrac{S_{a,f}-S_{A2,f}}{|S_{A2,f}|+\varepsilon},
& \text{для метрик, где больше лучше}.
\end{cases}
\]

Кандидат проходит отбор, только если одновременно выполнены условия:

\[
\operatorname{mean}_f g_{a,f}\ge 0,
\qquad
\operatorname{median}_f g_{a,f}\ge 0,
\]

\[
\frac{1}{K}\sum_f \mathbf{1}[g_{a,f}>0]\ge\frac{2}{3},
\qquad
CI^{95\%}_{\mathrm{bootstrap,lower}}(\bar g_a)\ge -0.005.
\]

Для regression аналогичная проверка неухудшения выполняется по `MAE`. Если A5 и A6 отклонены, решение получает статус `fallback_to_a2`. Если оба кандидата допустимы, выбирается вариант с наибольшей нижней границей доверительного интервала, затем с наибольшими медианным и средним выигрышем. Порядок входных записей на решение не влияет.

```bash
python examples/benchmark/rmt_cross_fitted_geometry_selector_experiment.py \
  --selection-folds 5 \
  --seeds 42 43 44 45 46 \
  --budgets 0.01 0.05 0.10 0.20
```

Первая серия ограничена датасетами `Brazilian_houses`, `elevators`, `adult`, `jannis`. Она создает 320 основных записей и три дополнительных артефакта:

- `cross_fitted_geometry_runs.jsonl` с одним решением на сочетание dataset/seed/budget/model;
- `geometry_selection_fold_losses.csv` с метриками каждого варианта на каждом внутреннем фолде;
- `geometry_selection_summary.csv` с bootstrap-интервалами, долей положительных фолдов и причинами отклонения кандидатов.

### 4.7. Phase A.3: A9 с контролем хвоста регрессионной ошибки

Итог Phase A.2 выявил различие между типичной и хвостовой ошибкой: в четырёх регрессионных конфигурациях A8 улучшил `MAE`, но проиграл A2 по `RMSE`. Поэтому A9 использует условное среднее крупнейших абсолютных ошибок:

\[
\operatorname{TailMAE}_q(y,\hat y)
=\frac{1}{k}\sum_{i\in\operatorname{TopK}(|y-\hat y|,k)}|y_i-\hat y_i|,
\qquad
k=\max\left(1,\left\lceil(1-q)n\right\rceil\right),
\quad q=0.9.
\]

`TailMAE` является эмпирическим аналогом CVaR для абсолютной ошибки. В A9 он вычисляется на каждом внутреннем фолде и становится обязательным ограничением только при основной метрике `rmse`. Кандидат отклоняется, если средний или медианный относительный выигрыш по `TailMAE` ниже `-0.005` либо нижняя граница 95%-го бутстрэп-доверительного интервала ниже этого порога.

Регрессионные внутренние фолды A9 формируются по десяти сбалансированным слоям рангов целевой переменной. Это не превращает целевую переменную в категориальную: слои используются только для равномерного распределения диапазона целей между отложенными частями фолдов.

```bash
python examples/benchmark/rmt_cross_fitted_geometry_selector_experiment.py \
  --tail-risk-guard \
  --tail-risk-quantile 0.90 \
  --tail-noninferiority-margin 0.005 \
  --regression-stratification-bins 10 \
  --selection-folds 5 \
  --seeds 42 43 44 45 46 \
  --budgets 0.01 0.05 0.10 0.20
```

По умолчанию Phase A.3 использует новые относительно A.2 задачи: `diamonds`, `OnlineNewsPopularity`, `bank-marketing`, `covertype`. В `geometry_selection_fold_losses.csv` сохраняются `tail_metric`, `tail_value`, `tail_quantile`; в `geometry_selection_summary.csv` — средний и медианный выигрыш, доля положительных фолдов и границы доверительного интервала хвостовой метрики.

## 5. Phase B: Bulk/Spike Hierarchical Experts

### 5.1. Спектральное разделение

Для сингулярных значений оценивается empirical null edge `lambda_edge` с помощью трех null policies:

- feature permutation;
- moment-matched Gaussian;
- view resampling.

На screening используются 16 resamples, на финальной проверке 32. Null quantile равен `0.95`, минимальная selection frequency устойчивой компоненты равна `0.80`.

Множество spike-компонент:

\[
J_{\mathrm{spike}}=\{j:\sigma_j>\lambda_{\mathrm{edge}}
\ \land\ f_j\ge 0.8\}.
\]

Для строки `i` вычисляются:

\[
e_i^{\mathrm{spike}}=
\sum_{j\in J_{\mathrm{spike}}}
(\sigma_j^2-\lambda_{\mathrm{edge}}^2)_+ U_{ij}^2,
\qquad
e_i^{\mathrm{bulk}}=
\sum_{j\notin J_{\mathrm{spike}}}\sigma_j^2U_{ij}^2,
\]

\[
g_i=\frac{e_i^{\mathrm{spike}}}
{e_i^{\mathrm{spike}}+e_i^{\mathrm{bulk}}+\varepsilon}.
\]

`g_i` определяет signalness строки, а нормированный вектор вкладов по `J_spike` задает ее spike signature. Если устойчивых spikes нет, система возвращает состояние `no_stable_spikes` и обучает только bulk expert: искусственно создавать spike partition запрещено.

### 5.2. Arms

| ID | Sampling / topology | Проверяемая гипотеза |
|---|---|---|
| B0 | Current `capped_leverage` + лучшая Phase A geometry | Воспроизводимый baseline. |
| B1 | Component-aware sampling, одна модель на объединенной выборке | Эффект только нового sampling без MoE topology. |
| B2 | Один bulk expert + один aggregate spike expert | Прямая проверка двухкомпонентной гипотезы пользователя. |
| B3 | Один bulk expert + несколько experts по spike signatures | Основная иерархическая RMT-гипотеза. |
| B4 | Обычные partition experts + лучшая Phase A geometry | Контроль topology без component split. |

Доля spike budget определяется excess spectral energy и ограничивается интервалом `[0.10, 0.50]`. До адаптивной политики screening сравнивает фиксированные доли `(0.10, 0.25, 0.40)`.

На основном сравнении bulk и spike experts используют одну model family и сопоставимую capacity. Более простая bulk model является отдельной последующей абляцией: иначе нельзя разделить эффект topology и model capacity.

### 5.3. Classification safety

- Каждая обучаемая classification partition должна содержать все глобальные классы, когда это допустимо бюджетом.
- Для каждого класса задается `min_class_count`; недостижимое ограничение дает `class_coverage_infeasible`.
- Repair rows входят в общий уникальный data budget.
- В diagnostics сохраняются class counts до и после repair, missing classes и class-distribution drift.

### 5.4. Gate bulk/spike-гипотезы

Основная гипотеза подтверждается только если B2 или B3:

1. превосходит B0 по median paired primary metric;
2. сохраняет преимущество при одинаковом `total_unique_train_rows`;
3. работает как минимум на двух бюджетах и на regression, и на classification;
4. не сводится к выбору почти всех строк одним expert;
5. показывает согласованность между spike stability, routing confidence и downstream expert advantage.

## 6. Статистический анализ

- Сравнение выполняется попарно внутри одинаковых `dataset/split/seed/budget/model`.
- Для median gain строится paired bootstrap 95% CI.
- Отдельно показываются median, worst-case и число выигранных datasets.
- Результаты агрегируются по task type и budget, но dataset-level таблица остается обязательной.
- Проверка множества routing arms считается exploratory; выбранные A- и B-гипотезы подтверждаются на отдельном holdout-наборе задач.
- Пропуски и failed runs не заменяются нулем и анализируются отдельно.

## 7. Артефакты

Phase A:

- `routing_geometry_runs.jsonl`;
- `routing_geometry_replay.csv`;
- `routing_geometry_summary.csv`;
- heatmap `metric x representation`;
- quality-budget curves;
- routing entropy, oracle regret и expert utilization plots.

Phase B:

- `bulk_spike_runs.jsonl`;
- `spectral_component_diagnostics.jsonl`;
- `row_participation_summary.csv`;
- `bulk_spike_summary.csv`;
- spectrum/null-edge plots;
- spike stability plots;
- budget allocation и expert utilization plots.

Каждая запись должна содержать `run_id`, git commit, dataset/task id, split/seed, budget, model, sampler config hash, routing geometry config, null policy, selected rank, stable spike indices, unique rows, overlap rows, status и structured failure reason.

## 8. Порядок запуска

1. [x] Реализовать contracts, matrix/Torch backend и invariant tests для Phase A.
2. [x] Выполнить synthetic routing smoke для regression и multiclass classification.
3. [x] Запустить real-data Phase A с LightGBM: `1120/1120`, `0 failed`.
4. [x] Реализовать независимый calibration/selection split и A7 selector contracts.
5. [x] Запустить real-data A7 selector gate и выявить чувствительность выбора к одному holdout-разбиению.
6. [x] Реализовать cross-fitted A8 selector и инкрементальные артефакты доказательств.
7. [x] Запустить малую реальную проверку A8 на четырёх датасетах: `320/320`, `0 failed`.
8. [x] Реализовать A9 с `TailMAE`, квантильными регрессионными фолдами и защитой возобновления запуска.
9. [ ] Запустить малую проверку A9 на новых датасетах и восстановить контрфактические решения A8 из оценок на тех же внутренних фолдах.
10. Реализовать component split и row participation contracts.
11. Выполнить synthetic bulk/spike gate.
12. Запустить B0-B4 screening с LightGBM.
13. Перепроверить только прошедшие gate варианты с TabPFN.
14. После этого расширять эксперимент на полную AMLB grid.

Полный benchmark нельзя запускать сразу после реализации: точкой первого повторного запуска является synthetic Phase A smoke, а первой значимой real-data серией является Phase A LightGBM screening.
