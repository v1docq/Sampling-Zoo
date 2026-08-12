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
| bank-marketing__task_359982 | 0.100000 | B0_standard_A9 | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.861776 | 0.064748 | 10.186093 | 0.054382 | 200.000000 | 16.000000 | - | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.822480 |
| bank-marketing__task_359982 | 0.100000 | B1_bulk_single_spike | 1 | 0.004391 | 0.004391 | 0.004391 | 1.000000 | 0.866167 | 0.060357 | 10.200834 | 0.027807 | 100.000000 | 18.000000 | - | -0.045839 | -0.009987 | 0.006909 | 0.017013 | - |
| bank-marketing__task_359982 | 0.100000 | B2_bulk_multi_spike | 1 | 0.004391 | 0.004391 | 0.004391 | 1.000000 | 0.866167 | 0.060357 | 10.497426 | 0.028145 | 100.000000 | 18.000000 | - | -0.045839 | -0.009987 | 0.006909 | 0.017013 | - |
| bank-marketing__task_359982 | 0.100000 | B3_validation_selected | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.861776 | 0.064748 | 10.186093 | 0.054382 | 200.000000 | 16.000000 | - | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.822480 |
| diamonds__task_233211 | 0.100000 | B0_standard_A9 | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1116.374760 | 1.003680 | 9.608641 | 0.034802 | 200.000000 | 14.000000 | 0.000000 | - | - | - | - | 0.716472 |
| diamonds__task_233211 | 0.100000 | B1_bulk_single_spike | 1 | -0.062174 | -0.062174 | -0.062174 | 0.000000 | 1185.784487 | 1.128257 | 9.663484 | 0.029815 | 200.000000 | 11.000000 | -0.121187 | - | - | - | - | 0.764544 |
| diamonds__task_233211 | 0.100000 | B2_bulk_multi_spike | 1 | -0.062174 | -0.062174 | -0.062174 | 0.000000 | 1185.784487 | 1.128257 | 9.956989 | 0.029487 | 200.000000 | 11.000000 | -0.121187 | - | - | - | - | 0.764544 |
| diamonds__task_233211 | 0.100000 | B3_validation_selected | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1116.374760 | 1.003680 | 9.608641 | 0.034802 | 200.000000 | 14.000000 | 0.000000 | - | - | - | - | 0.716472 |
