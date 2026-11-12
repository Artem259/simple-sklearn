# Changelog

## [0.2.0](https://github.com/Artem259/simple-sklearn/compare/v0.1.0...v0.2.0) (2026-11-12)


### ⚠ BREAKING CHANGES

* **classification:** redesign decision tree node hierarchy and typing:
    - Tree nodes no longer share a common base class; `DecisionTreeNode` is
    now a union type alias.
    - Node types in `ID3DecisionTreeClassifier` and `ID3SplitterNode` are
    narrowed to `ID3DecisionTreeNode`.
    - `ID3DecisionTreeClassifier.predict()` no longer raises `TypeError` for
    unknown node classes.
    - `LeafNode` and `ID3SplitterNode` are now dataclasses.
* **classification:** rename DecisionTreeClassifier to ID3DecisionTreeClassifier ([#12](https://github.com/Artem259/simple-sklearn/issues/12))

### Performance Improvements

* **agglomerative:** vectorize linkage distance calculations and optimize matrix updates ([#9](https://github.com/Artem259/simple-sklearn/issues/9)) ([6eac021](https://github.com/Artem259/simple-sklearn/commit/6eac021a97c091498bbd4d8f5d1f7006e48af529))


### Code Refactoring

* **classification:** extract base tree nodes and narrow ID3 node types ([#15](https://github.com/Artem259/simple-sklearn/issues/15)) ([5fcdc1e](https://github.com/Artem259/simple-sklearn/commit/5fcdc1e50ee2064c1aa65a5fd36523503894d3bf))
* **classification:** rename DecisionTreeClassifier to ID3DecisionTreeClassifier ([#12](https://github.com/Artem259/simple-sklearn/issues/12)) ([b384ce4](https://github.com/Artem259/simple-sklearn/commit/b384ce465cb939e6c16ba7e8bdaaed5b5fa636e3))

## 0.1.0 (2026-07-15)


* Initial release
