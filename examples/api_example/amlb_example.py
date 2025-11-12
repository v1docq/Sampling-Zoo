from core.utils.amlb_setup import LargeScaleAutoMLExperiment

EXPERIMENT_CONFIFG = {
    #'covtype-normalized': {'n_partitions': 10},
    'kddcup': {'n_partitions': 100}
}
# Запуск эксперимента
if __name__ == "__main__":
    experiment = LargeScaleAutoMLExperiment(experiment_config=EXPERIMENT_CONFIFG)
    experiment.run_full_benchmark()
    experiment.generate_report()
