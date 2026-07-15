# Changelog

## 0.1.0 (2026-07-15)


### Features

* initialize project with all files ([575b5d4](https://github.com/Artem259/simple-sklearn/commit/575b5d410333bc04b6291e558d66fb2e09dc5924))
* **kmedoids:** switch to alternate algorithm with custom convergence check and inertia ([ff641d7](https://github.com/Artem259/simple-sklearn/commit/ff641d7b5c5dbe9febeb4f741db1b3fa40cd3fb5))
* **typing:** configure package typing with mypy and annotate KNeighborsClassifier ([8b87e6e](https://github.com/Artem259/simple-sklearn/commit/8b87e6e3ec13b32c56cfb0d7bca98a8b7000f412))


### Bug Fixes

* **agglomerative:** ensure correct cluster labels assignment during fit ([2576104](https://github.com/Artem259/simple-sklearn/commit/2576104b77b176a191af4b401820207fc97e90e1))
* **base_partitional:** ensure non-string 'init' parameter is iterable ([51a52ef](https://github.com/Artem259/simple-sklearn/commit/51a52ef9b084c74f721750172caea5157579e744))
* **kmeans:** enforce strict shape validation for custom init array ([e39cec3](https://github.com/Artem259/simple-sklearn/commit/e39cec32ca857ff7273f6fc0b82fa9aae4e004ab))
* **typing:** correct _calc_log_probs return type in NaiveBayesClassifier ([6104087](https://github.com/Artem259/simple-sklearn/commit/6104087f343eb12d28d0b35cf79ecc2348e7c9b3))
* validate n_clusters against n_samples and standardize float formatting in errors ([95cf5d9](https://github.com/Artem259/simple-sklearn/commit/95cf5d967f302447a977cd175fb7fcd75e8d38f6))


### Miscellaneous Chores

* expand markdownlint configuration and migrate to .markdownlint-cli2.yaml ([#5](https://github.com/Artem259/simple-sklearn/issues/5)) ([1cdeccc](https://github.com/Artem259/simple-sklearn/commit/1cdeccc8167f31504d8142ae6b3fc129cbff7474))

## Changelog
