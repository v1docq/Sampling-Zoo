DEFAULT_BUDGET_RATIOS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)

AMLB_CATEGORY_PROFILES: dict[str, tuple[str, ...]] = {
    "small_samples_many_classes": ("amlb_optdigits", "amlb_vehicle"),
    "large_samples_binary": ("amlb_adult", "amlb_covertype"),
    "balanced_multiclass": ("amlb_mfeat_factors", "amlb_segment"),
}

