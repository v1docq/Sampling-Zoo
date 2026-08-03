# RMT Row-Selection Ablation

## Цель

Эксперимент отделяет качество кластеризации от качества выбора строк внутри уже
полученных RMT-партиций. Он отвечает на вопрос: является ли проигрыш RMT простым
baseline-методам следствием слишком агрессивного отбора строк с высоким leverage
score, если spectral embedding и состав исходных кластеров не меняются.

## Контролируемые факторы

Во всех сравниваемых запусках фиксированы:

- `cluster_selection_metric="balanced_silhouette"`;
- `view_strategy="gaussian"`;
- `router="spectral"`;
- `ensemble_method="routed_weighted"`;
- dataset split, seed, модель и `budget_ratio`.

Меняется только `selection_method`:

- `hybrid`: половина строк выбирается по leverage, остаток через greedy maxvol;
- `leverage`: строки с максимальными leverage scores;
- `maxvol`: геометрически разнообразные строки spectral embedding;
- `capped_leverage`: leverage scores ограничиваются локальным квантилем, после
  чего выполняется PPS-выборка без возвращения.

Для `capped_leverage` используется
`leverage_cap_quantile=0.95`. Ограничение вычисляется отдельно внутри каждой
партиции. Нулевые или некорректные веса дают равномерную выборку без возвращения.

## Инвариант партиций

Sampler записывает `partition_membership_fingerprint`: SHA-256 от канонической
последовательности cluster labels. Канонизация делает fingerprint устойчивым к
переименованию кластеров. Для одной пары `dataset x budget_ratio` fingerprint
должен совпадать у всех row-selection методов. Иначе сравнение смешивает два
фактора и не должно использоваться для выбора production policy.

## Запуск

Быстрый механизм-проверочный прогон:

```powershell
.\.venv_rmt_cuda39\Scripts\python.exe examples\benchmark\rmt_row_selection_ablation.py --mechanism-smoke
```

Он использует `Brazilian_houses`, `diamonds`, `pol`, бюджеты `1%`, `5%`, `20%`
и четыре row-selection метода. Полный запуск использует стандартную RMT-сетку
бюджетов:

```powershell
.\.venv_rmt_cuda39\Scripts\python.exe examples\benchmark\rmt_row_selection_ablation.py
```

## Артефакты и решение

Основной файл сравнения: `row_selection_comparison.csv`. В нем находятся:

- RMSE и `rmse_delta_vs_hybrid`;
- fit/inference time и их дельты;
- target drift подвыборки;
- `partition_fingerprint_count` и `partition_fingerprint_consistent`.

Новый default выбирается только если метод улучшает медианный RMSE на нескольких
датасетах и бюджетах, не создает тяжелого худшего случая, не ухудшает target drift
и сохраняет приемлемое время обучения. Один выигрыш на одном датасете не является
достаточным основанием для смены default.
