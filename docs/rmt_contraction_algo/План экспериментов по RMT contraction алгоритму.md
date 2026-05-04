#  Первичный план - 04.05.26

1. Прогнать `rmt_contraction` против `random`, `difficulty`, `feature_clustering` на 3–5 regression datasets из `AMBL_regression_suite.csv`.
2. Разделить результаты на:
    - маленькие датасеты -  `<20k` семплов;
    - средние датасеты `20k–200k` семплов;
    - больше датасеты `>200k` семплов.
3. Для каждого датасета построить кривую зависимости бюджета вычислений по `chunk_fraction`.
4. Сравнить два режима:
    - `ensemble_method="voting"`;
    - `ensemble_method="routed_weighted"`.