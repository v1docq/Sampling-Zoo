# Resume и идемпотентное выполнение RMT-эксперимента

## Назначение

Долгий benchmark может завершиться после нескольких часов из-за ошибки модели,
сбоя CUDA, потери соединения или остановки процесса. Повторный запуск не должен:

- заново обучать уже завершенные комбинации;
- создавать дубликаты в authoritative JSONL;
- смешивать результаты несовместимых конфигураций;
- менять исходный `run_id` и provenance запуска.

Resume-слой решает эту задачу на уровне минимального leaf-run:

```text
dataset x split x model x strategy x budget x router x view x seed
```

## Публичный интерфейс

```python
run_rmt_contraction_regression_experiment(
    resume_from="examples/benchmark/results/run_rmt_contraction_regression_...",
    resume_policy="retry_failed",
)
```

`resume_from` указывает на корневой каталог существующего запуска, содержащий
`artifact_manifest.json`, `run_meta.json` и raw JSONL.

Поддерживаются две политики:

| Политика | Завершенные leaf-run | Failed leaf-run |
|---|---|---|
| `retry_failed` | пропускаются | удаляются из authoritative JSONL и выполняются повторно |
| `skip_existing` | пропускаются | сохраняются и также пропускаются |

По умолчанию используется `retry_failed`.

## Pure Core

Модуль `sampling_zoo.core.experiment.resume` не читает файловую систему.
Он содержит:

- `LeafRunKey` - frozen-контракт канонической идентичности leaf-run;
- `ResumePolicy` и `LeafRunOutcome`;
- `ResumePlan` - решение о retained, skipped и retry записях;
- `build_resume_plan(...)` - чистое преобразование records в план;
- helpers для построения ключа из runtime-компонентов и исторической записи.

Ключом является SHA-256 канонического JSON payload. Полный payload сохраняется
рядом с digest в `extra.leaf_run`, поэтому запись остается проверяемой человеком.

Если один ключ встречается несколько раз, действуют правила:

1. completed имеет приоритет над failed;
2. среди completed сохраняется последняя запись;
3. среди failed сохраняется последняя запись;
4. число отброшенных дублей попадает в resume diagnostics.

Это защищает от сценария, когда успешный результат уже существовал, а более
поздняя аварийная попытка создала failed-record с тем же ключом.

## Effectful Shell

Модуль `sampling_zoo.core.experiment.resume_runtime` отвечает только за IO:

1. читает и типизированно валидирует `artifact_manifest.json`;
2. проверяет совпадение `run_id` с именем каталога;
3. сравнивает SHA-256 научной конфигурации;
4. находит root artifact с ролью `raw_runs`;
5. проверяет containment пути и hash файла;
6. читает JSONL и передает записи в pure `build_resume_plan(...)`.

Несовместимая конфигурация отклоняется через
`ResumeCompatibilityError(code="resume_config_mismatch")`.
В hash научной конфигурации не входят runtime-поля `resume_from`,
`resume_policy` и `show_progress`.

Для manifest старой версии, созданного до появления resume-полей, поддержан
legacy hash-кандидат. Это совместимость только с прежним способом вычисления
digest, а не разрешение менять модели, budgets, tasks или seed.

## Crash Recovery

JSONL дописывается до обновления manifest. Поэтому после жесткого завершения
процесса hash raw-файла может быть новее hash в manifest.

Правила восстановления:

- для `completed` hash mismatch всегда считается повреждением;
- для `running` и `failed` stale hash допустим и фиксируется в diagnostics;
- для `running` и `failed` можно отбросить только одну поврежденную последнюю
  непустую строку;
- поврежденная строка в середине JSONL всегда является ошибкой;
- для `completed` число parsed records обязано совпадать с `record_count`.

Перед повторным выполнением `IncrementalExperimentSaver.restore_records(...)`
атомарно переписывает raw JSONL retained-записями через временный файл,
`flush/fsync` и `os.replace`. Поэтому failed-запись не остается рядом с новой
успешной записью того же leaf-run.

`logs/strategy_runs.jsonl` является журналом событий и может содержать историю
повторных попыток. Authoritative набор результатов для отчетов находится в
`metrics/rmt_regression_runs.jsonl`.

## Интеграция с Runner

`EnsembleChunkBenchmarkRunner` получает `ResumePlan`, но не выполняет IO.
Он строит planned leaf-ключи и:

- не загружает OpenML split, если весь dataset/model/strategy grid завершен;
- передает план в `EnsembleFoldBenchmarkExecutor`;
- executor проверяет ключ до `run_single_fold`;
- новая запись всегда содержит `extra.leaf_run_key` и `extra.leaf_run`.

`RMTRegressionExperimentOrchestrator` владеет runtime lifecycle:

```text
load manifest and records
  -> validate compatibility
  -> mark original run as running
  -> atomically retain completed records
  -> execute failed and missing leaves
  -> rebuild reports
  -> finalize original run_id
```

Resume diagnostics сохраняются в `run_meta.json` в поле `resume`.

## Инварианты для тестов

- одинаковые оси leaf-run дают одинаковый ключ;
- изменение любой оси меняет ключ;
- `build_resume_plan` детерминирован;
- completed-запись не заменяется failed-записью;
- после resume authoritative JSONL не содержит повторяющихся leaf-ключей;
- несовместимый config hash не запускает обучение;
- полностью завершенный dataset не вызывает OpenML loading;
- forced interrupted smoke завершается после resume без повторного обучения
  уже сохраненных leaf-run.

## Text2Image Prompt

```text
ICML-style scientific figure, clean academic vector infographic, white background, muted blue-gray palette with one accent color, minimal typography, precise arrows, thin lines, labeled panels, no photorealism, no 3D glossy rendering, no decorative background, conference-paper figure aesthetics, mathematically clean, visually balanced. Show an idempotent experiment resume protocol in three labeled panels: first panel has a dataset-model-strategy-fold grid producing canonical SHA-256 leaf keys; second panel shows an interrupted append-only JSONL and a typed artifact manifest validation gate; third panel shows completed keys bypassing training, failed keys being atomically retried, and a final duplicate-free authoritative JSONL. Include pure decision core and effectful runtime shell as separate horizontal layers.
```
