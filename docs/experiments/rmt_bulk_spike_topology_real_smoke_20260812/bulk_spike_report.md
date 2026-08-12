# Пилот топологий тела и сигнала на реальных данных

Статус критерия допуска: **failed**.

## Критерии

| проверка | результат |
|---|---|
| all_records_completed | True |
| full_dataset_references_completed | True |
| specialized_topologies_preserve_exact_budget | True |
| median_gain_nonnegative | True |
| worst_harm_within_margin | True |
| selected_specialization_win_rate_at_least_80pct | False |
| regression_tail_noninferiority | True |
| classification_probability_and_balance_noninferiority | True |

## Сводка

| dataset | budget_ratio | arm_name | runs | mean_gain | median_gain | worst_gain | win_rate | mean_primary_value | mean_degradation_vs_full | mean_fit_seconds | mean_inference_seconds | mean_tree_count | mean_max_depth | mean_tail_gain | mean_brier_gain | mean_ece_gain | mean_f1_macro_gain | mean_worst_class_recall_gain | mean_shap_stability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_multiclass_topology | 0.200000 | B0_standard_A9 | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.537152 | 0.695091 | 0.379638 | 0.007121 | 0.000000 | 0.000000 | - | 0.000000 | 0.000000 | 0.000000 | 0.000000 | - |
| synthetic_multiclass_topology | 0.200000 | B1_bulk_single_spike | 1 | 0.218520 | 0.218520 | 0.218520 | 1.000000 | 0.419774 | 0.324680 | 0.376871 | 0.002931 | 0.000000 | 0.000000 | - | 0.166753 | -0.022122 | 0.025534 | 0.025000 | - |
| synthetic_multiclass_topology | 0.200000 | B2_bulk_multi_spike | 1 | 0.218520 | 0.218520 | 0.218520 | 1.000000 | 0.419774 | 0.324680 | 0.381854 | 0.002760 | 0.000000 | 0.000000 | - | 0.166753 | -0.022122 | 0.025534 | 0.025000 | - |
| synthetic_multiclass_topology | 0.200000 | B3_validation_selected | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.537152 | 0.695091 | 0.379638 | 0.007121 | 0.000000 | 0.000000 | - | 0.000000 | 0.000000 | 0.000000 | 0.000000 | - |
| synthetic_rmt_regression_smoke | 0.200000 | B0_standard_A9 | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 15.026625 | 0.119337 | 0.574321 | 0.004128 | 0.000000 | 0.000000 | 0.000000 | - | - | - | - | - |
| synthetic_rmt_regression_smoke | 0.200000 | B1_bulk_single_spike | 1 | -0.188226 | -0.188226 | -0.188226 | 0.000000 | 17.855030 | 0.330026 | 0.106726 | 0.002872 | 0.000000 | 0.000000 | -0.285242 | - | - | - | - | - |
| synthetic_rmt_regression_smoke | 0.200000 | B2_bulk_multi_spike | 1 | -0.188226 | -0.188226 | -0.188226 | 0.000000 | 17.855030 | 0.330026 | 0.130338 | 0.003584 | 0.000000 | 0.000000 | -0.285242 | - | - | - | - | - |
| synthetic_rmt_regression_smoke | 0.200000 | B3_validation_selected | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 15.026625 | 0.119337 | 0.574321 | 0.004128 | 0.000000 | 0.000000 | 0.000000 | - | - | - | - | - |
