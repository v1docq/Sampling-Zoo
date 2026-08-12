# Масштабирование качества RMT по бюджету

Аппроксимация использует функцию `loss(b) = L_full + A * (b^(-alpha) - 1)` с точным якорем опорной модели при `b=1`. Для ROC AUC используется потеря `1 - ROC AUC`.

Параметр `alpha` описывает скорость ухудшения при уменьшении бюджета. Высокий `R^2` на четырёх бюджетах и одной опорной точке является предварительным свидетельством, а не универсальным законом.

| dataset | model | arm_name | status | primary_metric | asymptotic_loss | scale | exponent | exponent_at_boundary | evidence_status | exponent_confidence_interval | r_squared | mean_absolute_error | reference_primary_value | reference_loss | required_budget_by_degradation | n_budget_points | n_observations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_multiclass_topology | ridge | B0_standard_A9 | fitted | log_loss | -9.652084 | 9.947436 | 0.010000 | True | boundary_limited | - | 0.828599 | 0.033294 | 0.295352 | 0.295352 | {"1pct": 0.9707494628622226, "3pct": 0.9148143185400748, "5pct": 0.8621325361621563} | 4 | 4 |
| synthetic_multiclass_topology | ridge | B3_validation_selected | fitted | log_loss | -9.652084 | 9.947436 | 0.010000 | True | boundary_limited | - | 0.828599 | 0.033294 | 0.295352 | 0.295352 | {"1pct": 0.9707494628622226, "3pct": 0.9148143185400748, "5pct": 0.8621325361621563} | 4 | 4 |
| synthetic_rmt_regression_smoke | ridge | B0_standard_A9 | fitted | rmse | 13.359860 | 0.010008 | 3.000000 | True | boundary_limited | - | 0.996984 | 1.050153 | 13.369868 | 13.369868 | {"1pct": 0.4114256448202471, "3pct": 0.28982320598255884, "5pct": 0.24524500344620415} | 4 | 4 |
| synthetic_rmt_regression_smoke | ridge | B3_validation_selected | fitted | rmse | 12.078519 | 1.291350 | 0.705788 | False | exploratory_sparse_grid | - | 0.956832 | 0.583294 | 13.369868 | 13.369868 | {"1pct": 0.8697187070565693, "3pct": 0.6816489920488469, "5pct": 0.5537297490934417} | 4 | 4 |
