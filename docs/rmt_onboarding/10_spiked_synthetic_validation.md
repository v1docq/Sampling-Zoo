# Spiked synthetic validation для RMT

## Зачем нужен отдельный synthetic benchmark

OpenML-эксперименты измеряют итоговое качество ансамбля, но не позволяют
однозначно установить источник ошибки. Плохой RMSE может быть следствием
spectral embedding, выбора числа partitions, кластеризации, routing или
chunk-моделей.

Spiked benchmark изолирует первый вопрос:

> восстанавливает ли RMT spectral engine известный низкоранговый сигнал при
> контролируемом отношении signal-to-noise?

Benchmark является диагностическим. Он не меняет production rank policy и не
использует известный истинный ранг при обучении sampler.

## Контракт данных

Чистый модуль
`sampling_zoo/core/validation/spiked_models.py` содержит immutable contracts:

- `SpikedDataConfig` валидирует размер матрицы, истинный ранг, SNR, noise family
  и seed;
- `SpikedDataset` хранит наблюдаемую матрицу, signal/noise decomposition,
  истинные left/right subspaces и эмпирический SNR;
- `SpikedValidationGridPoint` задает один воспроизводимый leaf-run;
- `RankRecoveryMetrics`, `SubspaceRecoveryMetrics` и
  `SpikedRecoveryEvaluation` описывают результаты без зависимости от IO.

Генерация, построение grid и расчет recovery-метрик являются чистыми
детерминированными функциями. Загрузка torch, запуск sampler, tqdm и запись
артефактов остаются в runtime shell
`examples/benchmark/run_rmt_spiked_synthetic_experiment.py`.

## Математическая модель

Наблюдаемая матрица строится как

\[
X = L + N,\qquad
L = U\operatorname{diag}(\sigma_1,\ldots,\sigma_r)V^\top,
\]

где \(U^\top U=I_r\), \(V^\top V=I_r\), а \(r\) является известным истинным
рангом. Столбцы \(U\) ортогональны константному вектору, поэтому centering в
табличном preprocessing не удаляет часть signal subspace.

Сингулярные значения сигнала линейно убывают от `1.0` до
`min_spike_ratio`. Реализация поддерживает:

- `gaussian`: независимый нормальный шум;
- `student_t`: стандартизованный тяжелохвостый шум с конечной дисперсией,
  поэтому `student_t_df > 2`.

После генерации шум центрируется и масштабируется так, чтобы для каждого
конкретного leaf-run выполнялось точное эмпирическое равенство

\[
\mathrm{SNR}_{F}
=
\frac{\lVert L\rVert_F^2}{\lVert N\rVert_F^2}.
\]

Это устраняет случайный разброс фактического SNR между seed и делает
NumPy/Torch сравнение парным: оба backend получают одну и ту же матрицу.

Сам sampler затем применяет `StandardScaler`. Диагональное масштабирование
признаков не меняет истинный left signal span, но меняет отношение энергий.
Поэтому рядом с точным исходным `empirical_snr` сохраняется
`standardized_empirical_snr`, рассчитанный после того же per-feature scaling.
При анализе detectability threshold следует проверять обе оси, особенно для
Student-t noise.

Для `true_rank=0` строится отдельный null control. Его шум имеет unit RMS, а
`snr=0`.

## Recovery ранга

Benchmark независимо сравнивает с истинным рангом четыре оценки:

| Оценка | Источник |
|---|---|
| `explained_estimated_rank` | текущий production explained-variance rank |
| `null_edge_estimated_rank` | число компонент выше empirical null edge |
| `view_stability_estimated_rank` | стабильные spectral outliers при новых views |
| `subspace_stability_estimated_rank` | максимальный устойчивый prefix span |

Для каждой оценки сохраняются signed/absolute error, precision, recall, F1,
exact-match и false-positive flag. На rank-zero controls false positive
означает любой положительный оцененный ранг.

Высокий explained-variance rank на чистом шуме ожидаем: эта политика измеряет
сохраненную энергию, а не статистическую отделимость сигнала. Поэтому ее нельзя
сравнивать с null-edge rank как две реализации одной и той же величины.

## Recovery подпространства

Пусть \(\widehat U\) — выбранный sampler left basis. Через singular values
\(c_i\) матрицы \(U^\top\widehat U\) вычисляется overlap:

\[
\operatorname{overlap}
=
\lVert U^\top\widehat U\rVert_F^2
=
\sum_i c_i^2.
\]

Метрики для неравных рангов:

\[
\operatorname{recall}_{subspace}
=
\frac{\operatorname{overlap}}{r},
\qquad
\operatorname{precision}_{subspace}
=
\frac{\operatorname{overlap}}{\widehat r}.
\]

`selected_subspace_recall` штрафует sampler за потерянные истинные направления,
а `selected_subspace_precision` — за лишние направления. F1 не позволяет
скрыть under-ranking хорошим principal angle на пересечении меньшей размерности.

Дополнительно сохраняются:

- mean/max principal angle;
- минимальная canonical correlation;
- `missed_subspace_distance = sqrt(1 - recall)`.

Все метрики инвариантны к знакам, перестановкам и внутреннему вращению базиса.

## Запуск

Минимальная локальная проверка:

```powershell
python examples\benchmark\run_rmt_spiked_synthetic_experiment.py --smoke
```

Стандартный grid:

```powershell
python examples\benchmark\run_rmt_spiked_synthetic_experiment.py
```

Полный server grid из 20 SNR, 10 seed, двух noise families и двух backend:

```powershell
python examples\benchmark\run_rmt_spiked_synthetic_experiment.py --server-grid
```

Server grid дорогой: null и subspace diagnostics выполняют дополнительные SVD
для каждого leaf-run. Перед полным запуском следует проверить smoke на том же
torch/CUDA окружении.

Публичный Python entrypoint:

```python
from examples.benchmark.run_rmt_spiked_synthetic_experiment import (
    run_rmt_spiked_synthetic_experiment,
)

output_dir = run_rmt_spiked_synthetic_experiment(
    snr_values=(0.1, 0.3, 1.0, 3.0, 10.0),
    seeds=range(5),
    backends=("numpy", "torch"),
)
```

## Артефакты

Каждый leaf-run сначала fsync-записывается в
`metrics/rmt_spiked_runs.jsonl`. Этот JSONL содержит полный вложенный sampler
diagnostics. Компактные таблицы, report и manifest атомарно обновляются
checkpoint-ами каждые `snapshot_every` записей и обязательно при нормальном
завершении:

- `metrics/rmt_spiked_raw_runs.csv`;
- `metrics/rmt_spiked_summary_by_snr.csv`;
- `metrics/rmt_spiked_null_controls.csv`;
- `metrics/rmt_spiked_backend_agreement.csv`;
- `report.md`;
- `run_meta.json`;
- `artifact_manifest.json`.

Падение позднего leaf-run не уничтожает предыдущие результаты. Недоступный
torch backend фиксируется как `skipped`; иная ошибка leaf-run — как `failed`,
после чего grid продолжает выполняться.

## Как принимать последующее решение

Перед изменением production rank policy следует проверить одновременно:

1. На rank-zero controls null/stability ranks имеют приемлемый false-positive
   rate при заданном `null_quantile`.
2. После detectability threshold растут rank recall и selected-subspace recall.
3. При высоком SNR rank diagnostics приближаются к известному рангу.
4. Парные NumPy/Torch результаты согласуются в пределах заранее выбранного
   tolerance.
5. Student-t degradation понятна и не маскируется усреднением с Gaussian runs.

Только после этих проверок разумно реализовывать явную комбинированную
production policy и `selected_rank_reason`.

## Ограничения первой версии

Первая версия покрывает один глобальный low-rank regime. Она еще не проверяет:

- mixture of low-rank regimes и cluster recovery;
- correlated или sparse noise;
- categorical perturbations после encoding;
- downstream sample efficiency и качество chunk-моделей.

Эти сценарии должны расширять тот же typed core, а не добавляться как
несвязанные ad-hoc scripts.

## Text2Image Prompt

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Four-panel scientific figure for controlled RMT spiked validation: panel A latent low-rank matrix L equals U diagonal spikes V transpose plus Gaussian or Student-t noise N with exact Frobenius SNR; panel B the same observed matrix enters random feature contractions and NumPy or Torch spectral backends; panel C four rank curves versus logarithmic SNR labeled explained variance, null edge, view stability, and subspace stability with the true rank as a horizontal reference; panel D rotation-invariant subspace precision and recall from canonical correlations, plus paired backend absolute deltas and rank-zero false-positive controls. Show equations for empirical SNR and squared subspace overlap, thin confidence bands across seeds, precise scientific labels.
```
