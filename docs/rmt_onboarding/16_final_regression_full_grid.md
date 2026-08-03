# Финальная сетка RMT regression

## Назначение

`examples/benchmark/rmt_regression_full_grid.py` является entrypoint для повторного
регрессионного эксперимента после завершения ablation-этапов. Он не строит декартово
произведение всех моделей и стратегий. Каждая строка плана явно связывает одну модель,
одну sampling strategy, способ обучения partition models и бюджет.

Это защищает эксперимент от бессмысленных комбинаций и позволяет однозначно
интерпретировать стоимость каждого сценария.

## Основная сетка

Бюджеты: `1%`, `5%`, `10%`, `20%` от train split.

Datasets по умолчанию:

- `diamonds`;
- `house_16H`;
- `house_sales`;
- `elevators`;
- `pol`;
- `Brazilian_houses`;
- `OnlineNewsPopularity`.

Train split ограничивается `100_000` строками. Это сохраняет сравнимость запусков и
удерживает concatenated TabPFN context на уровне не более `20_000` строк при бюджете
`20%`.

### LightGBM controls

- full-dataset baseline;
- `RMT`, `random`, `difficulty` с независимой моделью на каждом chunk;
- `RMT`, `random`, `difficulty` с одной моделью на конкатенации всех выбранных строк.

### TabPFN in-context

- независимые RMT experts с spectral routing;
- одна модель на concatenated RMT sample;
- concatenated random и difficulty controls.

### TabPFN fine-tuning

Повторяет четыре сценария TabPFN in-context, но обновляет веса через публичный
fine-tuning estimator. По умолчанию установлен предел `900` секунд на одну модель.

Полная матрица содержит `57` сценариев на dataset и `399` leaf runs для семи
datasets. Сценарии разделены на группы `lightgbm_controls`, `tabpfn_in_context` и
`tabpfn_finetuned`.

## Зафиксированный RMT profile

Финальная сетка не повторяет завершённые ablations. Для RMT используется следующий
профиль:

```python
{
    "view_strategy": "gaussian",
    "n_views": "auto",
    "embedding_mode": "sv_scaled",
    "partition_selection_method": "auto",
    "cluster_selection_metric": "balanced_silhouette",
    "cluster_ensemble_method": "coassociation",
    "selection_method": "capped_leverage",
    "leverage_cap_quantile": 0.95,
}
```

Независимые RMT experts используют `routed_weighted` и spectral router. В
concatenated режиме существует одна активная модель, поэтому применяется `voting` и
router не создаётся.

## Контракты сценариев

`ModelStrategyScenarioSpec` связывает:

- уникальное имя сценария;
- имя model factory;
- typed `StrategySpec`;
- группу запуска.

`ModelStrategyScenarioGridContract` валидирует уникальность имён и сохраняет порядок
моделей. `EnsembleChunkBenchmarkRunner.run_scenario_grid(...)` исполняет только эти
явные пары. Legacy dict материализуется лишь на границе с `SamplingEnsemble`.

В `strategy_params` сохраняются три идентификатора:

- `experiment_scenario` содержит конкретный бюджет;
- `scenario_family` остаётся одинаковым вдоль budget curve;
- `scenario_group` определяет крупный блок вычислений.

Агрегации sample efficiency используют `scenario_family`, поэтому результаты разных
бюджетов образуют одну кривую, но independent и concatenated режимы не смешиваются.

## Baselines и деградация

Для LightGBM reference является full-dataset модель на том же split. Для TabPFN,
если full-data запуск отсутствует, используется `foundational` RMSE из AMLB reference
table. Поле `rmse_ref_source` принимает значения:

- `full_dataset`;
- `amlb_foundational`;
- `best_observed`, если внешнего или full-data reference нет.

`rmse_drop = (rmse - rmse_ref) / rmse_ref`. Таблица
`minimal_effective_budget.csv` ищет минимальный бюджет при допустимой деградации
`1%`, `3%` и `5%`.

## Вычислительные и complexity diagnostics

Каждый run сохраняет:

- `fit_time`, `inference_time`;
- `fit_rows_per_second`, `inference_rows_per_second`;
- фактическое число строк, использованных всеми активными моделями;
- число активных chunk models;
- для LightGBM: суммарное число trees, leaves и splits, среднюю и максимальную
  глубину;
- normalized entropy gain importance;
- normalized entropy и top-5 share для mean absolute SHAP contributions.

Tree/SHAP diagnostics являются best-effort. Для TabPFN ставится
`complexity_status="unsupported"`; это не превращает валидный model run в ошибку.

В `plots/` формируются отдельные фигуры для каждой пары dataset/model:

- `degradation__*.png`;
- `runtime_quality__*.png`;
- `complexity__*.png`, если модель предоставляет tree diagnostics.

## Запуск

Сначала проверить CUDA и TabPFN в том же окружении, из которого будет запущен benchmark:

```bash
python -c "import torch, tabpfn; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Synthetic smoke без OpenML и TabPFN:

```bash
python examples/benchmark/rmt_regression_full_grid.py --smoke
```

Полный запуск:

```bash
python examples/benchmark/rmt_regression_full_grid.py
```

Целевой блок:

```bash
python examples/benchmark/rmt_regression_full_grid.py \
  --scenario-group tabpfn_in_context \
  --task diamonds
```

Resume должен использовать ту же конфигурацию, включая набор scenario groups и
fine-tuning параметры:

```bash
python examples/benchmark/rmt_regression_full_grid.py \
  --resume-from examples/benchmark/results/run_rmt_regression_full_grid_<timestamp>
```

Каждый leaf run записывается в JSONL до перехода к следующему сценарию. При падении
процесса уже завершённые строки сохраняются, а resume пропускает их по стабильному
`leaf_run_key`.

## Text2Image prompt

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Final RMT regression benchmark as a typed non-Cartesian scenario graph: seven AMLB regression datasets flow into four budget nodes (1, 5, 10, 20 percent), then explicit model-strategy bindings for LightGBM controls, TabPFN in-context, and TabPFN fine-tuning; distinguish independent routed experts from one concatenated model; show incremental leaf-run persistence, reference RMSE selection, degradation curves, runtime metrics, and tree/SHAP complexity diagnostics in labeled panels.
```
