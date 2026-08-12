# Масштабирование качества RMT по бюджету

Аппроксимация использует функцию `loss(b) = L_full + A * (b^(-alpha) - 1)` с точным якорем опорной модели при `b=1`. Для ROC AUC используется потеря `1 - ROC AUC`.

Параметр `alpha` описывает скорость ухудшения при уменьшении бюджета. Высокий `R^2` на четырёх бюджетах и одной опорной точке является предварительным свидетельством, а не универсальным законом.

| dataset | model | arm_name | status | primary_metric | asymptotic_loss | scale | exponent | exponent_at_boundary | evidence_status | exponent_confidence_interval | r_squared | mean_absolute_error | reference_primary_value | reference_loss | required_budget_by_degradation | n_budget_points | n_observations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_multiclass_topology | ridge | B0_standard_A9 | fitted | log_loss | -8.761051 | 9.077937 | 0.010000 | True | boundary_limited | - | 0.824466 | 0.030531 | 0.316887 | 0.316887 | {"1pct": 0.9657007682266618, "3pct": 0.9006241573991918, "5pct": 0.8399737815237218} | 4 | 4 |
| synthetic_multiclass_topology | ridge | B3_validation_selected | fitted | log_loss | -8.761051 | 9.077937 | 0.010000 | True | boundary_limited | - | 0.824466 | 0.030531 | 0.316887 | 0.316887 | {"1pct": 0.9657007682266618, "3pct": 0.9006241573991918, "5pct": 0.8399737815237218} | 4 | 4 |
| synthetic_rmt_regression_smoke | ridge | B0_standard_A9 | fitted | rmse | 13.414573 | 0.010001 | 3.000000 | True | boundary_limited | - | 0.996914 | 1.048835 | 13.424574 | 13.424574 | {"1pct": 0.41081650424930355, "3pct": 0.28937327756324993, "5pct": 0.24486054715101868} | 4 | 4 |
| synthetic_rmt_regression_smoke | ridge | B3_validation_selected | fitted | rmse | 12.180003 | 1.244571 | 0.714897 | False | exploratory_sparse_grid | - | 0.955844 | 0.586879 | 13.424574 | 13.424574 | {"1pct": 0.8665060904909365, "3pct": 0.6755981289588147, "5pct": 0.5469675682900158} | 4 | 4 |
