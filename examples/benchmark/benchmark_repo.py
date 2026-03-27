DEFAULT_BUDGET_RATIOS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)

ENSEMBLE_CV_FOLDS: int = 3
ENSEMBLE_N_PARTITIONS: int = 3
ENSEMBLE_STRATEGIES: tuple[str, ...] = ("difficulty", "random")
ENSEMBLE_MODELS: tuple[str, ...] = ("tabpfn", "lightgbm")
AMLB_CUSTOM_CLASSIFICATION_DATASETS: tuple[str, ...] = (
    "covtype-normalized",
    "kddcup",
    "sf-police-incidents",
)
AMLB_CUSTOM_REGRESSION_DATASETS: tuple[str, ...] = (
    "airlines",
    "year_prediction_msd",
)

AMLB_CATEGORY_PROFILES: dict[str, tuple[str, ...]] = {
    "small_samples_many_classes": (
        "amlb_optdigits",
        "amlb_vehicle",
        "amlb_mfeat_factors",
        "amlb_segment",
        "amlb_satimage",
        "amlb_letter",
    ),
    "large_samples_binary": (
        "amlb_adult",
        "amlb_covertype",
        "amlb_magic_telescope",
        "amlb_spambase",
        "amlb_banknote_authentication",
        "amlb_ionosphere",
    ),
    "tabular_mixed_classification": (
        "amlb_credit_g",
        "amlb_kr_vs_kp",
        "amlb_sick",
        "amlb_waveform",
        "amlb_phoneme",
        "amlb_page_blocks",
        "amlb_wine_quality_red",
        "amlb_wine_quality_white",
    ),
    "amlb_top20_mix": (
        "amlb_adult",
        "amlb_covertype",
        "amlb_optdigits",
        "amlb_vehicle",
        "amlb_mfeat_factors",
        "amlb_segment",
        "amlb_credit_g",
        "amlb_kr_vs_kp",
        "amlb_sick",
        "amlb_spambase",
        "amlb_letter",
        "amlb_satimage",
        "amlb_waveform",
        "amlb_phoneme",
        "amlb_page_blocks",
        "amlb_ionosphere",
        "amlb_banknote_authentication",
        "amlb_wine_quality_red",
        "amlb_wine_quality_white",
        "amlb_magic_telescope",
    ),
}
