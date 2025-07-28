```@meta
CurrentModule = DataSplits
```

# 03. Core API Reference

**Note:** DataSplits expects data matrices to follow the Julia ML convention: **columns are samples, rows are features**. All examples and API references below assume this convention. If you have data with samples as rows, transpose it before splitting (e.g., use `X'`).

For custom data types, implement `Base.length` (number of samples) and `Base.getindex(data, i)` (returning the i-th sample) as described in the [MLUtils documentation](https://juliaml.github.io/MLUtils.jl/). This ensures compatibility with all DataSplits algorithms and utilities.

All split strategies in DataSplits return indices in the range `1:N` (where `N` is the number of samples). For non-standard arrays or custom containers, always use `getobs(X, indices)` to access the split data, as this will correctly handle any custom indexing or axes.

## split

```julia
split(data, strategy; rng)
```

Dispatches to `strategy._split`. Returns a `SplitResult` object representing the split.

## SplitResult API

The result of a split is always a subtype of the abstract type `SplitResult`. This API provides a structured, type-safe way to represent train/test (and validation) splits, as well as cross-validation folds.

### Abstract Type

```julia
abstract type SplitResult end
```

### Concrete Subtypes

#### TrainTestSplit

```julia
struct TrainTestSplit{I} <: SplitResult
    train::I
    test::I
end
```

- `train`: indices of training samples
- `test`: indices of test samples

#### TrainValTestSplit

```julia
struct TrainValTestSplit{I} <: SplitResult
    train::I
    val::I
    test::I
end
```

- `train`: indices of training samples
- `val`: indices of validation samples
- `test`: indices of test samples

#### CrossValidationSplit

```julia
struct CrossValidationSplit{T<:SplitResult} <: SplitResult
    folds::Vector{T}
end
```

- `folds`: a vector of `TrainTestSplit` or `TrainValTestSplit` objects, one per fold

## splitdata

```julia
splitdata(result::SplitResult, X)
```

Given a `SplitResult` and data `X`, returns the corresponding splits as a tuple. For example:

- For `TrainTestSplit`, returns `(X_train, X_test)`
- For `TrainValTestSplit`, returns `(X_train, X_val, X_test)`
- For `CrossValidationSplit`, returns a vector of tuples, one per fold

## Example Usage

```julia
using DataSplits

result = split(X, KennardStoneSplit(0.8))
X_train, X_test = splitdata(result, X)

result = split(X, ClusterStratifiedSplit(clusters, :equal; n=4, frac=0.7))
X_train, X_test = splitdata(result, X)

# Cross-validation
cv_result = split(X, SomeCVSplit(...))
for (X_train, X_test) in splitdata(cv_result, X)
    # ...
end
```

## Indices returned by split strategies

All split strategies in DataSplits return indices in the range `1:N` (where `N` is the number of samples). For non-standard arrays or custom containers, always use `getobs(X, indices)` to access the split data, as this will correctly handle any custom indexing or axes. This approach is fully compatible with the MLUtils interface and ensures robust, generic code for all data types.

## SplitStrategy Interface

To add a new strategy, subtype `SplitStrategy` and implement:

```julia
_split(data, s::YourStrategy; rng)
```

## Utility Functions

- `sample_indices(data)`: returns iterable indices of samples (default: `1:size(data,1)`, but now robust to any AbstractArray axes).
- `ValidFraction`: bounds-checked fraction type for split ratios.

## Example: Custom Splitter (new pattern)

```julia
struct MySplit <: SplitStrategy; frac; end

function mysplit(N, s, rng, data)
    cut = floor(Int, s.frac * N)
    train_pos = 1:cut
    test_pos = (cut+1):N
    return train_pos, test_pos
end

function _split(data, s::MySplit; rng=Random.default_rng())
    N = numobs(data)
    train_pos, test_pos = mysplit(N, s, rng, data)
    return TrainTestSplit(train_pos, test_pos)
end
```
