from examples.experiments.amlb_setup import LargeScaleAutoMLExperiment

# Запуск эксперимента
if __name__ == "__main__":
    experiment = LargeScaleAutoMLExperiment()
    experiment.run_full_benchmark()
    experiment.generate_report()