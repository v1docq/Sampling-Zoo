# Leverage SHAP и RMT-семплирование строк

Статус критерия допуска: **не пройден**.

Эксперимент разделяет два разных применения leverage-оценок: выбор строк для обучения и выбор коалиций признаков для SHAP-регрессии.

## Общий результат

| passed | checks | informative_pair_count | median_leverage_gain | mean_leverage_gain | leverage_win_rate |
|---|---|---|---|---|---|
| False | {'all_records_completed': True, 'row_budgets_exact': True, 'projected_efficiency_holds': True, 'informative_median_leverage_gain_positive': False, 'informative_mean_leverage_gain_positive': False, 'informative_leverage_win_rate_at_least_60pct': False} | 299 | -0.000878 | -0.000643 | 0.481605 |

## Попарное сравнение способов выбора коалиций

| scenario | row_arm | budget_multiplier | median_leverage_gain | win_rate |
|---|---|---|---|---|
| diabetes_hist_gradient_boosting | D0_full_rows | 2.000000 | 0.068094 | 0.600000 |
| diabetes_hist_gradient_boosting | D0_full_rows | 4.000000 | -0.010061 | 0.400000 |
| diabetes_hist_gradient_boosting | D0_full_rows | 8.000000 | -0.002828 | 0.400000 |
| diabetes_hist_gradient_boosting | D0_full_rows | 16.000000 | 0.001364 | 0.500000 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | 2.000000 | -0.000000 | 0.500000 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | 4.000000 | -0.000000 | 0.400000 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | 8.000000 | -0.000000 | 0.450000 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | 16.000000 | 0.000000 | 0.600000 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | 2.000000 | -0.101202 | 0.350000 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | 4.000000 | -0.000000 | 0.450000 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | 8.000000 | -0.000000 | 0.350000 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | 16.000000 | 0.000000 | 0.700000 |
| synthetic_interaction | not_applicable | 2.000000 | 0.067140 | 0.650000 |
| synthetic_interaction | not_applicable | 4.000000 | -0.021062 | 0.250000 |
| synthetic_interaction | not_applicable | 8.000000 | 0.003946 | 0.750000 |
| synthetic_interaction | not_applicable | 16.000000 | -0.001575 | 0.400000 |
| synthetic_threshold | not_applicable | 2.000000 | 0.008897 | 0.500000 |
| synthetic_threshold | not_applicable | 4.000000 | -0.075860 | 0.400000 |
| synthetic_threshold | not_applicable | 8.000000 | 0.034226 | 0.700000 |
| synthetic_threshold | not_applicable | 16.000000 | 0.003187 | 0.550000 |

## Сводка по всем осям

| scenario | row_arm | coalition_policy | budget_multiplier | median_relative_error | mean_relative_error | worst_relative_error | median_efficiency_residual | median_runtime_seconds | median_downstream_rmse | median_explanation_drift | median_row_gram_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diabetes_hist_gradient_boosting | D0_full_rows | kernel_weight | 2.000000 | 0.243148 | 0.281795 | 0.682414 | 0.000000 | 0.009742 | 58.555430 | 0.000000 | 0.000000 |
| diabetes_hist_gradient_boosting | D0_full_rows | leverage | 2.000000 | 0.160003 | 0.226680 | 0.670609 | 0.000000 | 0.009704 | 58.555430 | 0.000000 | 0.000000 |
| diabetes_hist_gradient_boosting | D0_full_rows | kernel_weight | 4.000000 | 0.061976 | 0.079528 | 0.181891 | 0.000000 | 0.009658 | 58.555430 | 0.000000 | 0.000000 |
| diabetes_hist_gradient_boosting | D0_full_rows | leverage | 4.000000 | 0.087242 | 0.100235 | 0.330343 | 0.000000 | 0.010034 | 58.555430 | 0.000000 | 0.000000 |
| diabetes_hist_gradient_boosting | D0_full_rows | kernel_weight | 8.000000 | 0.040712 | 0.043675 | 0.091444 | 0.000000 | 0.009967 | 58.555430 | 0.000000 | 0.000000 |
| diabetes_hist_gradient_boosting | D0_full_rows | leverage | 8.000000 | 0.043210 | 0.048411 | 0.107830 | 0.000000 | 0.010160 | 58.555430 | 0.000000 | 0.000000 |
| diabetes_hist_gradient_boosting | D0_full_rows | kernel_weight | 16.000000 | 0.021994 | 0.024947 | 0.049610 | 0.000000 | 0.010203 | 58.555430 | 0.000000 | 0.000000 |
| diabetes_hist_gradient_boosting | D0_full_rows | leverage | 16.000000 | 0.024537 | 0.025162 | 0.049354 | 0.000000 | 0.011197 | 58.555430 | 0.000000 | 0.000000 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | kernel_weight | 2.000000 | 0.126579 | 0.209193 | 0.669696 | 0.000000 | 0.009992 | 62.152419 | 0.754644 | 0.193961 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | leverage | 2.000000 | 0.076789 | 0.157572 | 0.712814 | 0.000000 | 0.009781 | 62.152419 | 0.754644 | 0.193961 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | kernel_weight | 4.000000 | 0.000000 | 0.000574 | 0.007544 | 0.000000 | 0.009828 | 62.152419 | 0.754644 | 0.193961 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | leverage | 4.000000 | 0.000000 | 0.024359 | 0.468824 | 0.000000 | 0.009471 | 62.152419 | 0.754644 | 0.193961 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | kernel_weight | 8.000000 | 0.000000 | 0.000402 | 0.006326 | 0.000000 | 0.009235 | 62.152419 | 0.754644 | 0.193961 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | leverage | 8.000000 | 0.000000 | 0.000350 | 0.004520 | 0.000000 | 0.009818 | 62.152419 | 0.754644 | 0.193961 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | kernel_weight | 16.000000 | 0.000000 | 0.000203 | 0.003036 | 0.000000 | 0.009964 | 62.152419 | 0.754644 | 0.193961 |
| diabetes_hist_gradient_boosting | D1_uniform_rows | leverage | 16.000000 | 0.000000 | 0.000231 | 0.003851 | 0.000000 | 0.010776 | 62.152419 | 0.754644 | 0.193961 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | kernel_weight | 2.000000 | 0.048547 | 0.143397 | 0.663323 | 0.000000 | 0.009334 | 60.856493 | 0.740987 | 0.188522 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | leverage | 2.000000 | 0.173573 | 0.184316 | 0.647464 | 0.000000 | 0.010186 | 60.856493 | 0.740987 | 0.188522 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | kernel_weight | 4.000000 | 0.000000 | 0.001691 | 0.009396 | 0.000000 | 0.009670 | 60.856493 | 0.740987 | 0.188522 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | leverage | 4.000000 | 0.000000 | 0.028343 | 0.522765 | 0.000000 | 0.009618 | 60.856493 | 0.740987 | 0.188522 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | kernel_weight | 8.000000 | 0.000000 | 0.000824 | 0.004381 | 0.000000 | 0.009351 | 60.856493 | 0.740987 | 0.188522 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | leverage | 8.000000 | 0.000000 | 0.001212 | 0.007350 | 0.000000 | 0.010668 | 60.856493 | 0.740987 | 0.188522 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | kernel_weight | 16.000000 | 0.000000 | 0.000522 | 0.003608 | 0.000000 | 0.010763 | 60.856493 | 0.740987 | 0.188522 |
| diabetes_hist_gradient_boosting | D2_robust_leverage_ipw | leverage | 16.000000 | 0.000000 | 0.000691 | 0.004709 | 0.000000 | 0.010412 | 60.856493 | 0.740987 | 0.188522 |
| synthetic_interaction | not_applicable | kernel_weight | 2.000000 | 0.384108 | 0.379549 | 0.680294 | 0.000000 | 0.002648 | - | 0.000000 | - |
| synthetic_interaction | not_applicable | leverage | 2.000000 | 0.311473 | 0.360944 | 0.845428 | 0.000000 | 0.002690 | - | 0.000000 | - |
| synthetic_interaction | not_applicable | kernel_weight | 4.000000 | 0.092181 | 0.102380 | 0.180246 | 0.000000 | 0.002596 | - | 0.000000 | - |
| synthetic_interaction | not_applicable | leverage | 4.000000 | 0.112782 | 0.138643 | 0.509968 | 0.000000 | 0.002910 | - | 0.000000 | - |
| synthetic_interaction | not_applicable | kernel_weight | 8.000000 | 0.053956 | 0.058834 | 0.114546 | 0.000000 | 0.002667 | - | 0.000000 | - |
| synthetic_interaction | not_applicable | leverage | 8.000000 | 0.054586 | 0.052727 | 0.070763 | 0.000000 | 0.002986 | - | 0.000000 | - |
| synthetic_interaction | not_applicable | kernel_weight | 16.000000 | 0.030856 | 0.031748 | 0.048326 | 0.000000 | 0.002922 | - | 0.000000 | - |
| synthetic_interaction | not_applicable | leverage | 16.000000 | 0.030163 | 0.032147 | 0.046755 | 0.000000 | 0.003287 | - | 0.000000 | - |
| synthetic_threshold | not_applicable | kernel_weight | 2.000000 | 0.592050 | 0.651125 | 1.605181 | 0.000000 | 0.002506 | - | 0.000000 | - |
| synthetic_threshold | not_applicable | leverage | 2.000000 | 0.522902 | 0.636391 | 1.769517 | 0.000000 | 0.002674 | - | 0.000000 | - |
| synthetic_threshold | not_applicable | kernel_weight | 4.000000 | 0.235720 | 0.272374 | 0.506345 | 0.000000 | 0.002633 | - | 0.000000 | - |
| synthetic_threshold | not_applicable | leverage | 4.000000 | 0.285147 | 0.305152 | 0.426978 | 0.000000 | 0.002934 | - | 0.000000 | - |
| synthetic_threshold | not_applicable | kernel_weight | 8.000000 | 0.171108 | 0.166095 | 0.219719 | 0.000000 | 0.002871 | - | 0.000000 | - |
| synthetic_threshold | not_applicable | leverage | 8.000000 | 0.138830 | 0.142116 | 0.206333 | 0.000000 | 0.002940 | - | 0.000000 | - |
| synthetic_threshold | not_applicable | kernel_weight | 16.000000 | 0.103754 | 0.104731 | 0.156172 | 0.000000 | 0.003087 | - | 0.000000 | - |
| synthetic_threshold | not_applicable | leverage | 16.000000 | 0.095508 | 0.097515 | 0.147411 | 0.000000 | 0.003133 | - | 0.000000 | - |

Положительный `median_leverage_gain` означает меньшую относительную ошибку Leverage SHAP при том же числе вызовов модели.
