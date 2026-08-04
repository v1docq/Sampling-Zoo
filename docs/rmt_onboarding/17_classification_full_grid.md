# Финальная сетка RMT classification

## Назначение

`examples/benchmark/rmt_classification_full_grid.py` переносит проверенный регрессионный протокол на классификацию. Эксперимент остается probability-first: chunk-модели возвращают вероятности, `SamplingEnsemble` выравнивает их к глобальному `classes_`, а labels вычисляются только после ансамблирования.

Runner использует typed non-Cartesian grid. Каждая строка `ModelStrategyScenarioSpec` явно связывает модель, sampler, topology, router, class-allocation policy и бюджет. Legacy dict создается только на границе с `SamplingEnsemble`.

## Основная сетка

Бюджеты от train split:

- `1%`;
- `5%`;
- `10%`;
- `20%`.

AMLB/OpenML suite: `271`.

Задачи по умолчанию:

- `adult`;
- `bank-marketing`;
- `covertype`;
- `jannis`.

Train split стратифицированно ограничивается `50_000` строками. Если dataset является OpenML task, используется его официальный test split и один внутренний validation split. Поэтому LightGBM gate содержит `41` leaf run на dataset и `164` leaf runs для четырех datasets. Synthetic smoke использует два CV-fold и сохраняет `44` записи для двух datasets при одном бюджете.

## LightGBM gate

Для каждого бюджета запускаются:

- random sample, одна модель на concatenated sample;
- difficulty sample, одна модель на concatenated sample;
- RMT global capped: concatenated, voting, spectral routing, constrained gating;
- RMT class-stratified capped: concatenated, voting, spectral routing, constrained gating.

Дополнительно один раз обучается full-dataset LightGBM baseline на том же split.

### Global capped leverage

`class_allocation_policy="minimum_then_global"` резервирует не менее `min_samples_per_class` строк каждого локального класса. Остаток точного partition budget заполняется capped leverage по всем оставшимся строкам partition.

Преимущество: спектральная информативность сильнее влияет на итоговый sample. Риск: при сильном class imbalance распределение отобранных классов может дрейфовать.

### Class-stratified capped leverage

`class_allocation_policy="proportional"` строит точные квоты классов с lower bound и ограничением емкостью локального класса. Capped leverage затем применяется отдельно внутри каждой квоты.

Преимущество: лучше сохраняется локальное class distribution и проще интерпретировать drift. Риск: жесткие квоты уменьшают свободу спектрального отбора, особенно при бюджете `1%`.

Обе политики сохраняют точный общий budget. Infeasible plan не расширяет sample постфактум, а возвращает явную violation.

## Метрики

Primary metric выбирается на уровне dataset:

- binary classification: `roc_auc`, higher is better;
- multiclass classification: `log_loss`, lower is better.

Вторичные quality и calibration metrics:

- `accuracy`;
- `f1_macro`;
- `f1_weighted`;
- `brier_score`;
- `expected_calibration_error`.

Для binary Brier используется вероятность positive class. Для multiclass Brier считается средняя сумма квадратов отклонений полной probability matrix от one-hot target. ECE использует confidence победившего класса и десять equal-width bins.

`score_drop` всегда положителен при ухудшении:

```text
ROC AUC:  score_ref - score
log loss: score - score_ref
```

Minimal effective budget вычисляется отдельно для каждой scenario family при допустимом абсолютном ухудшении `0.01`, `0.03` и `0.05`. ROC AUC и log loss никогда не агрегируются в одну числовую метрику.

## Class-safety и routing diagnostics

Для RMT сохраняются:

- class counts до и после отбора;
- квоты классов;
- missing local/global classes;
- single-class chunk flag;
- class-distribution total-variation drift;
- точность budget plan и violations;
- строки, добавленные post-hoc repair;
- выбранное число partitions;
- validation/test routing entropy.

Допустимо, что multiclass chunk не содержит все глобальные классы, если исходный spectral partition их не содержал. Недопустим single-class chunk: такая partition не дает валидного локального classification expert.

## Артефакты

Каждый leaf run сразу записывается в:

```text
metrics/rmt_classification_full_grid_runs.jsonl
```

Snapshots и финализация создают:

- `classification_raw_runs.csv`;
- `classification_efficiency_curve.csv`;
- `classification_minimal_effective_budget.csv`;
- `classification_class_safety.csv`;
- quality/degradation plots;
- Brier/ECE calibration plots;
- class drift/single-class plots;
- `report.md` и `run_meta.json`.

## Порядок запуска

### 1. Synthetic smoke

```powershell
python examples/benchmark/rmt_classification_full_grid.py `
  --smoke `
  --scenario-group lightgbm_gate
```

### 2. Полный LightGBM gate

```powershell
python examples/benchmark/rmt_classification_full_grid.py `
  --scenario-group lightgbm_gate
```

Условия допуска к следующему этапу:

- завершены все `164` leaf runs;
- binary datasets имеют конечный ROC AUC;
- multiclass datasets имеют конечный log loss;
- RMT не создает single-class chunks;
- hard budget plan feasible либо содержит объяснимую structured violation;
- full-dataset baseline присутствует на каждом dataset;
- global/stratified capped сравниваются на одинаковых split и budget.

### 3. TabPFN in-context screening

TabPFN не включен по умолчанию. После прохождения LightGBM gate запускается отдельная группа:

```powershell
python examples/benchmark/rmt_classification_full_grid.py `
  --scenario-group tabpfn_in_context
```

Она содержит full-data baseline, random concatenated control и только class-stratified RMT: concatenated, spectral routing и constrained gating. Difficulty и global capped исключены, чтобы дорогое сравнение отвечало на конкретный вопрос о переносе лучшей class-safe sampling policy на foundation model.

### 4. Resume

```powershell
python examples/benchmark/rmt_classification_full_grid.py `
  --scenario-group lightgbm_gate `
  --resume-from examples/benchmark/results/run_rmt_classification_full_grid_<timestamp>
```

Resume должен использовать тот же набор tasks, budgets и scenario groups. Идентичность leaf run включает dataset, split, model и experiment scenario.

## Интерпретация первого запуска

Сначала сравниваются global и stratified capped в concatenated topology. Это изолирует качество самого sample от variance независимых experts и шума router. Затем сравниваются voting, spectral и constrained gating. Если stratified policy уменьшает class drift, но ухудшает primary metric, квоты слишком жесткие; если routing проигрывает concatenated при сопоставимом sample, следующая зона риска находится в expert specialization/router, а не в row selection.

## Text2Image prompt

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. A typed non-Cartesian RMT classification experiment: four AMLB datasets flow through budgets 1, 5, 10, and 20 percent; compare random and difficulty controls with global capped leverage and class-stratified capped leverage; show concatenated model, independent voting experts, spectral router, and constrained gating; binary branch ends in ROC AUC, multiclass branch ends in log loss, both branches emit Brier score, calibration error, class-drift diagnostics, exact-budget checks, and incremental artifacts.
```
