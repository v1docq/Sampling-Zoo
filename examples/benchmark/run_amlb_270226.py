from run_260226 import run_bench_pipeline

if __name__ == "__main__":
    run_bench_pipeline(include_amlb=True,full_benchmark=False,amlb_categories=("balanced_multiclass",
                                                                               "small_samples_many_classes"))
