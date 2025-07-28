```@meta
CurrentModule = DataSplits
```

# DataSplits

DataSplits is a Julia library for rational train/test splitting algorithms. It provides a variety of strategies for splitting datasets. In several applications random selection is not an appropriate choice and may lead to overestimating model performance.

| Strategy | Purpose | Complexity |
|----------|---------|------------|
| `KennardStoneSplit` | Maximin split on *X* | `O(N²)` time, `O(N²)` memory |
| `LazyKennardStoneSplit` | Same, streamed | `O(N²)` time, `O(N)` mem |
| `SPXYSplit` | Joint *X–y* maximin (SPXY) | `O(N²)` time, `O(N²)` mem |
| `OptiSimSplit`         | Optimisable dissimilarity-based splitting       | `O(N²)` time, `O(N²)` memory |
| `MinimumDissimilaritySplit`|  Greedy dissimilarity with one candidate | O(N²) time, O(N²) memory |
| `MaximumDissimilaritySplit`|  Greedy dissimilarity with full pool | O(N²) time, O(N²) memory |
| `ClusterShuffleSplit`|  Cluster-based shuffle split | O(N²) time, O(N²) memory |
| `ClusterStratifiedSplit`|  Cluster-based stratified split (equal, proportional, Neyman). Selects a quota per cluster, then splits into train/test according to user fraction. | O(N²) time, O(N²) memory |

All splitting strategies in DataSplits are designed to work with any AbstractArray, including those with non-standard axes.

**DataSplits expects data matrices to follow the Julia ML convention: columns are samples, rows are features.** If your data uses rows as samples, transpose it before splitting (e.g., use `X'`).

For custom data types, implement `Base.length` (number of samples) and `Base.getindex(data, i)` (returning the i-th sample) as described in the [MLUtils documentation](https://juliaml.github.io/MLUtils.jl/stable/api/). This ensures compatibility with all DataSplits algorithms and utilities.

## SplitResult API

All split operations return a `SplitResult` object, which provides a structured, type-safe way to represent train/test (and validation) splits, as well as cross-validation folds. Use the `splitdata` function to extract the actual data splits from a `SplitResult`.

### Example

```julia
julia> using DataSplits, Distances

julia> result = split(X, KennardStoneSplit(0.8))
TrainTestSplit
  train: [1, 2, ...]
  test: [101, 102, ...]

julia> X_train, X_test = splitdata(result, X)

julia> cv_result = split(X, SomeCVSplit(...))
CrossValidationSplit
  folds: [TrainTestSplit(...), ...]

julia> for (X_train, X_test) in splitdata(cv_result, X)
           # ...
       end
```
