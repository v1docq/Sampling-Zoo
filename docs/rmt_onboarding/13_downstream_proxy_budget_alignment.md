# Downstream proxy: alignment абсолютного бюджета

## Обнаруженная проблема

Первый mechanism smoke `run_rmt_partition_selection_ablation_20260803_135527`
показал аномалию на `Brazilian_houses` при бюджете 1%: downstream proxy выбрал
`k=1` и получил test RMSE `114290.27`.

Внешний selector проверял candidate на полном train из 7216 строк и выделял 72
строки. Внутренний holdout оставлял 80% train, после чего ошибочно повторно считал
`1%` уже от 5772 строк и получал только 58 строк. Поэтому допустимый внешний
candidate `k=2` с allocation `38 + 34` внутри proxy становился недопустимым:
для двух chunk-моделей требовалось минимум `32 + 32` строки.

## Исправленный контракт

Внутренний holdout меняет множество доступных строк, но не меняет абсолютный
budget будущего final fit:

\[
B = \operatorname{round}(N_{full\ train}\rho).
\]

`PartitionDownstreamProxyEvaluator` теперь передает в budget planner исходный
`N_full train`. Allocations строятся по размерам внутренних train-partitions, но
их сумма равна тому же `B`, который применит sampler на final fit.

В diagnostics добавлены:

- `budget_reference_rows`;
- `proxy_train_rows`;
- `requested_budget_size`;
- `selected_budget_size`;
- `budget_violations`.

## Контрольный rerun

Точечный запуск пересчитал только downstream objective:

```powershell
.\.venv_rmt_cuda39\Scripts\python.exe examples\benchmark\rmt_partition_selection_ablation.py --downstream-proxy-smoke
```

Результаты: `run_rmt_partition_selection_ablation_20260803_142121`.

| Dataset | Budget | k до | k после | RMSE до | RMSE после | Изменение |
|---|---:|---:|---:|---:|---:|---:|
| Brazilian houses | 1% | 1 | 2 | 114290.27 | 52042.47 | -54.5% |
| Brazilian houses | 5% | 3 | 3 | 24950.90 | 24950.90 | 0.0% |
| Brazilian houses | 20% | 3 | 3 | 13522.64 | 13522.64 | 0.0% |
| Diamonds | 1% | 2 | 2 | 1508.71 | 1508.71 | 0.0% |
| Diamonds | 5% | 2 | 2 | 968.45 | 968.45 | 0.0% |
| Diamonds | 20% | 2 | 2 | 682.25 | 682.25 | 0.0% |
| Pol | 1% | 2 | 2 | 29.47 | 29.47 | 0.0% |
| Pol | 5% | 2 | 2 | 25.12 | 25.12 | 0.0% |
| Pol | 20% | 3 | 3 | 10.92 | 10.92 | 0.0% |

Изменился только сценарий с ложной internal infeasibility. Это подтверждает, что
правка не меняет уже согласованные candidate decisions.

## Решение по production policy

`balanced_silhouette` остается default. Downstream proxy остается явной ablation:

- он выигрывает отдельные сценарии на `pol`;
- почти совпадает с balanced policy на `diamonds`;
- не дает устойчивого преимущества на `Brazilian_houses`;
- знак internal gain относительно concatenated baseline не всегда согласуется с
  test RMSE.

Следующая зона исследования находится ниже partition selector: при малых бюджетах
`hybrid` row selection может чрезмерно концентрироваться на leverage/outlier rows.
Это видно по сильному target drift на `Brazilian_houses`. До полной benchmark grid
нужно отдельно сравнить `hybrid`, `leverage`, `maxvol` и robust capped-leverage при
фиксированных partitions.
