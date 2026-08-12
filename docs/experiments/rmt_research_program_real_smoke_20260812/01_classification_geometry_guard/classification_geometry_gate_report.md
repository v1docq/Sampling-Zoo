# Критерий безопасности геометрии роутинга для классификации

Статус: **passed**.
Характер свидетельства: `safe_and_effective`.

Проверка использует ROC AUC для бинарных задач и log loss для многоклассовых задач. Безопасность и эффективность небазового выбора оцениваются раздельно.

## Проверки

- `expected_record_count`: `True`.
- `all_records_completed`: `True`.
- `expected_selector_record_count`: `True`.
- `primary_metric_contract`: `True`.
- `class_coverage_guaranteed`: `True`.
- `metric_alignment`: `True`.
- `selected_policy_noninferior`: `True`.

## Результаты по датасетам

| dataset | test_primary_metric | runs | nonbaseline_selections | mean_gain_vs_a2 | median_gain_vs_a2 | worst_gain_vs_a2 |
|---|---|---|---|---|---|---|
| adult__task_359983 | roc_auc | 1 | 0 | 0.000000 | 0.000000 | 0.000000 |
| car__task_359960 | log_loss | 1 | 0 | 0.000000 | 0.000000 | 0.000000 |
