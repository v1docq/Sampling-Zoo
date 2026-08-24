# Budget-aware partition selection для RMT

## Зачем потребовалось изменение

Раньше RMT sampler выбирал число кластеров по геометрии полного spectral embedding, а
`SamplingEnsemble` применял `budget_ratio` уже после построения partitions. При этом каждая
партиция урезалась отдельной случайной выборкой. Поэтому selector мог выбрать хорошее
разбиение при полном объеме данных, которое становилось неустойчивым при бюджете 1%, а
финальная выборка не использовала рассчитанные leverage scores.

Теперь бюджет является частью решения о partitions:

1. `SamplingEnsemble` передает `budget_ratio` в `RMTContractionTensorSampler` как
   `sampling_budget_ratio`.
2. Каждый кандидат получает детерминированный `PartitionBudgetPlan`.
3. Hard policy отклоняет кандидаты, для которых бюджет не обеспечивает минимальный размер
   всех локальных моделей или нарушает ограничения баланса.
4. Выбранный план повторно используется при формировании финальных chunks.
5. Строки внутри chunk выбираются методом `leverage`, `maxvol` или `hybrid`; повторного
   случайного урезания в `SamplingEnsemble` нет.

Это обеспечивает одинаковую семантику бюджета в selector, sampler и benchmark report.

## Контракты

`PartitionBudgetPlan` является immutable-контрактом между оценкой кандидата и семплированием.
Он хранит исходные размеры partitions, точные allocations, запрошенный общий бюджет и
нарушения ограничений. Сумма allocations равна доступному глобальному бюджету, если его
можно распределить в пределах исходных partitions.

Основные ограничения:

- `min_sampled_rows_per_partition=32`;
- `budget_max_imbalance_ratio=5.0`;
- `budget_min_partition_fraction=0.05`;
- `include_single_partition_candidate=True` как допустимый fallback для очень малого бюджета.

`PartitionSizeDiagnosticsContract` фиксирует размеры до и после применения бюджета,
количество выбранных и уникальных строк. `RuntimeDiagnosticsContract` отделяет время
preprocessing, spectral basis, cluster selection, chunk training и routing/finalization.

## Сравниваемые objectives

| Objective | Использует target | Учитывает бюджет | Что измеряет |
|---|---:|---:|---|
| `balanced_silhouette` | частично | нет hard-фильтра | Геометрию embedding с penalties за tiny/imbalanced clusters |
| `validation_proxy` | да | нет hard-фильтра | Выигрыш локальных константных experts на общем holdout |
| `budget_aware_validation_proxy` | да | да | Тот же дешевый proxy после проверки реализуемости бюджета |
| `downstream_proxy` | да | да | RMSE реальных budgeted chunk-моделей со spectral routing |

Для `downstream_proxy` сначала строятся все clustering candidates, затем по
`balanced_silhouette` выбирается shortlist. Для каждого кандидата на одном и том же
внутреннем holdout:

1. общий train budget распределяется между partitions;
2. строки выбираются тем же методом, что будет использовать sampler;
3. обучается одна облегченная копия downstream-модели на каждый chunk;
4. predictions смешиваются мягкими spectral routing probabilities;
5. дополнительно обучается одна модель на конкатенации тех же выбранных строк.

Итоговый score равен относительному выигрышу routed experts относительно глобального
среднего с небольшим штрафом за число experts. В отчете отдельно сохраняются RMSE routed
candidate, глобального среднего и concatenated budget baseline. Последний особенно важен:
он показывает, дает ли разбиение пользу сверх самого факта выбора хороших строк.

## Первый повторный запуск

Сначала запускается mechanism smoke, а не полная сетка:

```powershell
.\.venv_rmt_cuda39\Scripts\python.exe examples\benchmark\rmt_partition_selection_ablation.py --mechanism-smoke
```

Gate содержит:

- datasets: `Brazilian_houses`, `diamonds`, `pol`;
- budgets: `0.01`, `0.05`, `0.20`;
- четыре objectives из таблицы;
- один `LightGBM` model family, один seed.

Это 36 leaf-runs. Он проверяет основной механизм на малом, среднем и более крупном наборе,
а также на жестком, среднем и умеренном бюджете. Полную сетку из семи datasets и пяти
бюджетов следует запускать только после прохождения этого gate.

## Критерии перехода к полной сетке

1. Все leaf-runs завершены или имеют явный `failed/skipped` status без потери предыдущих
   результатов.
2. `selected_rows == requested_budget_size`, `duplicate_rows == 0`.
3. Для budget-aware objectives выбранный кандидат имеет `feasible=true`.
4. `downstream_proxy_candidate_loss` сравнивается одновременно с
   `downstream_proxy_concatenated_loss` и итоговым test RMSE.
5. Улучшение не определяется одним dataset или одним бюджетом.
6. Время cluster selection остается приемлемым относительно chunk training; иначе уменьшается
   `downstream_proxy_shortlist_size`, но не меняется семантика objective.

## Интерпретация отрицательного результата

- Хороший internal proxy и плохой test RMSE указывают на переобучение выбора partitions на
  внутренний holdout.
- Routed candidate хуже concatenated baseline означает, что локальные experts или routing не
  оправдывают partitioning при данном бюджете.
- Частый выбор `k=1` при 1% является валидным выводом feasibility policy, а не ошибкой: данных
  может быть недостаточно для нескольких устойчивых chunk-моделей.
- Высокий test gain при плохом proxy требует проверки leakage, split identity и точного
  соответствия budget allocations.

## Text2Image prompt

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Four-panel diagram of budget-aware RMT partition selection: panel A spectral embedding generates clustering candidates with k equals one through k max; panel B exact global budget contract allocates rows across partitions and rejects tiny or imbalanced candidates; panel C shortlist candidates train lightweight local experts on leverage-selected rows and combine predictions with soft spectral routing, alongside a concatenated-budget baseline; panel D report compares internal routed RMSE, concatenated RMSE, final test RMSE, partition sizes before and after budget, duplicate row count, and stage runtime. Highlight one shared budget plan flowing unchanged through selector, sampler, and ensemble.
```
