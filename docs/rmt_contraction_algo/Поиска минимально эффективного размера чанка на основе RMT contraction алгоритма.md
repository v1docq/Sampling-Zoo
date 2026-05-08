
# Подзадача: “насколько маленьким может быть чанк”

Эту задачу надо формализовать не как “найти хорошее число”, а как **sample efficiency curve**.

Для регрессии -  $m^*(\delta) = \min_m \left\{ m: RMSE_{\text{sampler}}(m) \le (1+\delta) RMSE_{\text{ref}} \right\}$

Где:

- m — общий training budget, управляемый `budget_ratio`;
- $\delta$ — допустимая деградация, например 1%, 3%, 5%;
- $RMSE_{\text{ref}}$​ — бейзлайн. Варианты бейзлайна могут быть следующие:
    - либо foundational model на всех данных;
    - либо текущая реализация ансамбля на random/difficulty семплерах;
    - либо лучший AutoML из `regression.csv`.

## Практическая реализация

В основной ветке `run_rmt_contraction_regression` сейчас оставлена одна главная ось бюджета:

```python
budget_ratio = [0.1, 0.3, 0.5, 0.75, 0.9]
```

`chunk_fraction` больше не используется как параллельная основная ось, потому что она дублировала смысл `budget_ratio` и усложняла интерпретацию. Ее лучше держать как отдельную RMT ablation: она отвечает за размер subset внутри уже найденного cluster-а, а не за общий budget всего ensemble.

Далеем строим таблицу:
```
dataset | sampler | ensemble_method | budget_ratio | total_train_rows | rmse | rmse_drop | fit_time | inference_time
```

Где критерий успеха:

```
минимальный budget_ratio, при котором RMSE не хуже baseline более чем на δ
```

В `RMTReportTableBuilder` это соответствует таблицам:

- `rmt_raw_runs.csv`;
- `sample_efficiency_curve.csv`;
- `minimal_effective_budget.csv`.

После последних изменений в raw/efficiency таблицы также попадают RMT diagnostics axes: `view_strategy`, `n_views`, `n_views_policy`, `selected_rank`, `selected_n_partitions`, `selected_cluster_algorithm`, а также EM routing refinement columns.

# Главные риски

## Риск 1. Метод может быть хуже random семплирования на “простых” датасетах

Если данные почти линейны, хорошо предобработаны и без выраженных режимов, random может оказаться не хуже. Это не провал метода, а сигнал, что tensor-contraction структура не даёт дополнительной информации.
## Риск 2. Категориальные признаки могут доминировать

Если one-hot даёт тысячи бинарных колонок, случайные feature views могут начать ловить артефакты кодирования, а не структуру данных. Поэтому в реализацию rmt семплера добавлены:

```
max_one_hot_cardinality
max_encoded_features
```

## Риск 3. RMT-обоснование не равно гарантии качества AutoML

RMT даёт контроль над спектральной структурой, но AutoML/foundational models могут быть чувствительны к другим вещам:

- распределению target;
- редким категориям;
- выбросам;
- предобработке признаков;

Поэтому обязательно нужны не только посчитанные метрики RMSE, но и результаты "диагностики методов":

```
sampler.diagnostics_
```

Там уже сохраняются:

```
mode0_unfolding_shape
n_views
n_views_policy
spectrum_stability_change
singular_values
selected_rank
leverage_entropy
effective_sample_count
selected_n_partitions
selected_cluster_algorithm
partition_selection_scores
chunk_sizes
```

Дополнительно для интерпретации sample efficiency нужно смотреть:

- partition diagnostics: target mean/std/quantiles per chunk, drift vs global, chunk size imbalance;
- routing diagnostics: entropy, mean max probability, validation/test assignment counts;
- routing refinement diagnostics, если включен `routing_refinement="em_retraining"`.
