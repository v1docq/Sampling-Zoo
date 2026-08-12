# Синтетический Phase S: leverage-preserving семплирование

- Статус: `passed`
- Завершено запусков: `252/252`
- Точный бюджет во всех запусках: `True`
- Повторяющиеся строки отсутствуют: `True`

## Общий рейтинг политик

| arm_id                |   mean_gain |   median_gain |   mean_gram_error |   mean_projection_error |
|:----------------------|------------:|--------------:|------------------:|------------------------:|
| S5_robust_mixture_ipw |    0.041720 |      0.022616 |          0.200733 |                0.064142 |
| S6_ridge_leverage_ipw |   -0.026384 |     -0.005302 |          0.288757 |                0.072064 |
| S2_capped_leverage    |   -0.098476 |      0.001692 |          0.289937 |                0.069247 |
| S4_robust_mixture     |   -0.256426 |     -0.012417 |          0.200733 |                0.064142 |
| S3_saturated_leverage |   -0.334587 |     -0.044609 |          0.192362 |                0.071102 |
| S1_top_leverage       |   -0.552314 |     -0.278931 |          0.617988 |                0.025305 |

## Кандидаты для Phase S на реальных данных

| arm_id                |   primary_gain_mean |   primary_gain_median |   gram_error_delta_mean |   worst_scenario_gain |
|:----------------------|--------------------:|----------------------:|------------------------:|----------------------:|
| S5_robust_mixture_ipw |            0.041720 |              0.022616 |               -0.185160 |              0.004666 |

## Эффекты по сценариям

| scenario                  | arm_id                |   mean_gain |   mean_gram_error |
|:--------------------------|:----------------------|------------:|------------------:|
| heavy_tail_regression     | S1_top_leverage       |   -0.440317 |          0.726297 |
| heavy_tail_regression     | S2_capped_leverage    |    0.000329 |          0.245669 |
| heavy_tail_regression     | S3_saturated_leverage |   -0.071610 |          0.249143 |
| heavy_tail_regression     | S4_robust_mixture     |   -0.046117 |          0.262722 |
| heavy_tail_regression     | S5_robust_mixture_ipw |    0.004666 |          0.262722 |
| heavy_tail_regression     | S6_ridge_leverage_ipw |    0.003982 |          0.292805 |
| high_leverage_regression  | S1_top_leverage       |   -1.361477 |          0.296656 |
| high_leverage_regression  | S2_capped_leverage    |   -0.377542 |          0.507489 |
| high_leverage_regression  | S3_saturated_leverage |   -1.230395 |          0.105379 |
| high_leverage_regression  | S4_robust_mixture     |   -0.950719 |          0.138620 |
| high_leverage_regression  | S5_robust_mixture_ipw |    0.146643 |          0.138620 |
| high_leverage_regression  | S6_ridge_leverage_ipw |   -0.070774 |          0.363951 |
| low_rank_regression       | S1_top_leverage       |   -0.430628 |          0.724499 |
| low_rank_regression       | S2_capped_leverage    |   -0.036102 |          0.203296 |
| low_rank_regression       | S3_saturated_leverage |   -0.052802 |          0.207463 |
| low_rank_regression       | S4_robust_mixture     |   -0.040861 |          0.200795 |
| low_rank_regression       | S5_robust_mixture_ipw |    0.005013 |          0.200795 |
| low_rank_regression       | S6_ridge_leverage_ipw |   -0.043920 |          0.249136 |
| rare_class_classification | S1_top_leverage       |    0.023164 |          0.724499 |
| rare_class_classification | S2_capped_leverage    |    0.019410 |          0.203296 |
| rare_class_classification | S3_saturated_leverage |    0.016461 |          0.207463 |
| rare_class_classification | S4_robust_mixture     |    0.011994 |          0.200795 |
| rare_class_classification | S5_robust_mixture_ipw |    0.010557 |          0.200795 |
| rare_class_classification | S6_ridge_leverage_ipw |    0.005177 |          0.249136 |

## Диагностические контроли для реальных данных

| arm_id                |   primary_gain_mean |   primary_gain_median |   gram_error_delta_mean |   worst_scenario_gain |
|:----------------------|--------------------:|----------------------:|------------------------:|----------------------:|
| S6_ridge_leverage_ipw |           -0.026384 |             -0.005302 |               -0.097136 |             -0.070774 |
| S2_capped_leverage    |           -0.098476 |              0.001692 |               -0.095955 |             -0.377542 |

## Интерпретация

Положительный gain означает меньший RMSE для регрессии или больший ROC AUC для классификации относительно равномерного семплирования при том же начальном значении генератора и бюджете.
Ошибки Gram-матрицы и стоимости проекции оценивают сохранение полной геометрии обучающей выборки до обучения итоговой модели.
Результат `S5_robust_mixture_ipw` следует интерпретировать как эффект согласованной пары: смешанная leverage-вероятность включения и обратное взвешивание при обучении. Политика `S4_robust_mixture` использует те же выбранные строки, но без коррекции весов.
