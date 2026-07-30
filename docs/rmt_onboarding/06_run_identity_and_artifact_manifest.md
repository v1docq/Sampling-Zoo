# Run Identity и manifest артефактов RMT-эксперимента

## Зачем нужен этот слой

Долгий RMT-бенчмарк должен оставлять воспроизводимый результат даже при
аварийном завершении. Одних CSV и `report.md` для этого недостаточно:
необходимо знать точную конфигурацию, версию кода, программное окружение,
CUDA-устройство, OpenML-задачи и назначение каждого файла.

Эту задачу решают два связанных контракта:

- `RunIdentity` фиксирует неизменяемую идентичность запуска;
- `ExperimentArtifactManifest` описывает текущее состояние запуска и реестр
  созданных артефактов.

JSON Schema v1 находится в
`docs/contracts/rmt_artifact_manifest_v1.schema.json`.

## Архитектурная граница

Реализация следует паттерну pure core / effectful shell.

### Чистое ядро

Модуль `sampling_zoo.core.experiment.artifact_manifest` содержит frozen
dataclasses, проверки инвариантов, построение, сериализацию и разбор manifest.
Он не читает файловую систему, не вызывает Git и не импортирует torch.

Основные типы:

- `RunIdentity`;
- `ExperimentConfigRef`;
- `AcceleratorIdentity`;
- `DatasetRunRef`;
- `RunArtifactRef`;
- `RunArtifactRegistry`;
- `ExperimentArtifactManifest`;
- `ArtifactManifestParseFailure`.

Ошибки в ожидаемых входных данных представлены структурированными
нарушениями `ArtifactManifestViolation`. Реестр не допускает два файла с
одинаковым путем или две корневые записи с одинаковой ролью.

### Runtime shell

Модуль `sampling_zoo.core.experiment.artifact_runtime` выполняет эффекты:

- получает commit и dirty-state из Git;
- фиксирует Python, версии ключевых пакетов, torch и CUDA;
- вычисляет SHA-256 конфигурации и файлов;
- обнаруживает известные артефакты в каталоге запуска;
- атомарно заменяет `artifact_manifest.json`.

Sampler, ensemble и numerical backend не знают о файловом manifest. Эта
ответственность находится на границе experiment runtime.

## Жизненный цикл

`RMTRegressionExperimentOrchestrator` подключает materializer к
`IncrementalExperimentSaver`. Поскольку classification и EM orchestrators
наследуют этот lifecycle, отдельная реализация для каждого runner не нужна.

Manifest обновляется:

1. после создания начального `run_meta.json`, статус `running`;
2. после каждого инкрементального снапшота;
3. после успешного завершения, статус `completed`;
4. после перехваченного исключения, статус `failed`.

Сначала атомарно записывается metadata, затем строится manifest. Поэтому
manifest никогда не ссылается на предыдущую версию `run_meta.json`.
Ошибка materializer не уничтожает уже сохраненные raw records: она попадает
в `metrics/incremental_saver_errors.jsonl`.

## Содержимое RunIdentity

`RunIdentity` создается один раз на запуск и затем не меняется:

- `run_id` и UTC-время старта;
- Git commit и признак dirty worktree;
- версия Python и платформа;
- версии Sampling Zoo, NumPy, pandas, SciPy, scikit-learn, OpenML,
  LightGBM, TabPFN и torch, если пакеты установлены;
- версия torch, CUDA runtime, доступность CUDA, число и имя GPU;
- SHA-256 эффективной experiment config;
- seed, row cap, OpenML suite IDs и запрошенные task names.

Полная конфигурация остается в `run_meta.json` внутри `experiment_plan`.
Manifest хранит ее digest и ключевые поля, чтобы быстро обнаруживать
несовпадение запусков.

## OpenML-провенанс

Каждая fold-запись теперь переносит:

- `suite_id`;
- `task_id` и `task_name`;
- `dataset_id`;
- `openml_repeat`, `openml_fold`, `openml_sample`;
- внутренний `split_label`.

Materializer группирует эти записи в `DatasetRunRef`. Таким образом,
`fold_0` внутренней кросс-валидации не смешивается с OpenML fold: это два
разных уровня разбиения.

## Реестр артефактов

Для каждого известного файла фиксируются:

- семантическая `role`;
- относительный POSIX path;
- SHA-256 и размер;
- stage-производитель;
- обязательность;
- имя и версия схемы для структурированных файлов;
- scope для множественных однотипных отчетов и таблиц.

При наличии run records обязательны ровно по одному корневому
`run_metadata` и `raw_runs`. Сам manifest не включает собственный hash:
это исключает рекурсивную зависимость.

## Как расширять

Новый тип артефакта добавляется декларативно через
`ArtifactDeclaration` в runtime-модуле. Для множественных файлов одной роли
нужно включить `scoped=True`, иначе инвариант уникальности отклонит manifest.

При несовместимом изменении структуры:

1. добавить новую константу версии и отдельную schema;
2. сохранить parser предыдущей версии или реализовать явную миграцию;
3. добавить round-trip, determinism и invalid-input tests;
4. не менять молча смысл существующих ролей.

## Проверка результата

После smoke run в каталоге эксперимента должны существовать:

```text
run_meta.json
artifact_manifest.json
metrics/rmt_*_runs.jsonl
```

Минимальные инварианты:

- `run_meta.run_identity.run_id == artifact_manifest.run.run_id`;
- `record_count` равен числу сохраненных run records;
- SHA-256 зарегистрированных файлов совпадают с фактическими;
- `completed` ставится только после финализации;
- повторная materialization без изменения входов дает те же байты.
