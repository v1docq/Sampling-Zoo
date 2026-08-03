# RMT classification sampling gate

## Зачем нужен отдельный gate

Регрессионный эксперимент не проверяет два критичных для классификации свойства:

1. После бюджетного отбора каждый обучающий chunk должен содержать как минимум два класса.
2. Binary ROC AUC и multiclass log loss должны считаться по вероятностям, выровненным к единому набору классов.

Поэтому перед полной сеткой регрессионных экспериментов запускается короткий classification gate. Его задача не выбрать окончательную конфигурацию, а проверить корректность sampling и inference contracts.

## Class-aware row selection

`RMTContractionTensorSampler` поддерживает параметры:

- `class_coverage_policy="auto" | "off" | "preserve_local_classes"`;
- `min_samples_per_class=1`.

В режиме `preserve_local_classes` отбор внутри каждого спектрального кластера выполняется в два шага:

1. Выбранная row-selection стратегия резервирует не менее `min_samples_per_class` строк каждого локально наблюдаемого класса.
2. Оставшийся точный бюджет заполняется той же стратегией из еще не выбранных строк.

План считается infeasible, если бюджет меньше числа обязательных class representatives или исходный кластер является single-class. Строки не добавляются постфактум, поэтому размер chunk остается равен budget allocation.

Диагностика `class_coverage_by_partition` хранит class counts до и после отбора, missing classes, total-variation drift, feasibility и violations.

## Probability-first ensemble

Для classification публичным inference contract является `SamplingEnsemble.ensemble_predict_proba(...)`.

- Вероятности каждой chunk-модели выравниваются к `SamplingEnsemble.classes_`.
- Отсутствующий в конкретной модели класс получает почти нулевую вероятность.
- `voting`, `weighted` и `routed_weighted` смешивают вероятности, а label получается через `argmax` после агрегации.
- Модель без `predict_proba` завершает run структурированной ошибкой `classification_probabilities_required`; hard-label fallback запрещен.

Основная AMLB-метрика выбирается по числу классов:

- binary: `roc_auc`, вероятность positive class;
- multiclass: `log_loss`, полная probability matrix.

## Runner

Entry point: `examples/benchmark/rmt_classification_sampling_gate.py`.

Synthetic smoke:

```powershell
python -c "from examples.benchmark.rmt_classification_sampling_gate import run_rmt_classification_sampling_gate; run_rmt_classification_sampling_gate(models=('random_forest',), synthetic_smoke=True)"
```

Реальный gate использует OpenML suite 271 и задачи `adult`, `credit-g`, `segment`. Сетка содержит full-data baseline, random voting и class-aware `RMT + capped_leverage` для voting/routed inference на бюджетах 5% и 20%.

## Условия допуска к полной сетке

- Ни один RMT chunk не является single-class.
- Размеры RMT chunks совпадают с точным budget plan.
- Binary runs содержат конечный ROC AUC.
- Multiclass runs содержат конечный log loss.
- Все classification модели предоставляют `predict_proba`.
- Incremental JSONL, CSV, metadata и Markdown report формируются без ошибок.
