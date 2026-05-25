```@meta
CurrentModule = DataSplits
```

# Getting Started

## Basic API

DataSplits exposes a single entry point, `partition`, in three forms:

```julia
# Two cohorts (train / test).
res = partition(data, alg; train, test, kwargs...)

# Three cohorts (train / validation / test) — two strategies.
res = partition(data, test_alg, val_alg; train, validation, test, kwargs...)

# Cross-validation — returns a CrossValidationSplit of folds.
cvs = partition(data, cv_alg; kwargs...)
```

`alg` is any subtype of [`AbstractSplitStrategy`](@ref); CV strategies subtype [`AbstractCVStrategy`](@ref). Cohort sizes (`train`, `validation`, `test`) live on `partition`, not on the strategy struct.

## Cohort sizes

`train`, `validation`, and `test` accept either absolute integer counts (summing to `numobs(data)`), integer percentages (summing to `100`), or `(0, 1)` fractions (summing to `1.0`).

```julia
partition(X, RandomSplit(); train = 80, test = 20)        # percentages
partition(X, RandomSplit(); train = 150, test = 40)       # absolute (requires numobs(X) == 190)
partition(X, RandomSplit(); train = 0.8, test = 0.2)      # fractions
```

CV strategies do not take `train`/`test` keywords — fold sizes are determined by the strategy itself (typically through `k` or `n_splits`). Resampling CV strategies (subtypes of [`AbstractResamplingCVStrategy`](@ref) such as `ShuffleSplit`) are the exception: they do accept `train`/`test`, applied per fold.

## Data formats

`data` can be a matrix, a vector, a Tables.jl-compatible container (e.g. `DataFrame`), or any custom type implementing the `MLUtils` interface (`MLUtils.numobs`, `MLUtils.getobs`).

For matrices DataSplits follows the Julia ML convention: **columns are samples, rows are features**. Transpose row-major data before passing it in (`X'` or `permutedims(X)`). For Tables.jl inputs rows are samples — DataSplits converts internally when a strategy needs an F×N matrix.

All returned indices are positive integers in `1:N`. For non-standard arrays or custom containers, materialise subsets with `splitdata` / `splitview` rather than indexing directly.

## Auxiliary slots

Strategies declare which auxiliary inputs they consume via the `consumes` trait. The three slots are:

- `target` — response / property vector (e.g. for `SPXYSplit`, `StratifiedKFold`).
- `time` — temporal ordering vector (e.g. for `TimeSplit`, `TimeSeriesSplit`, `BlockedCV`).
- `groups` — group-membership vector (e.g. for `GroupShuffleSplit`, `GroupKFold`).

```julia
partition(X, SPXYSplit(); target = y, train = 80, test = 20)
partition(X, GroupKFold(5); groups = patient_ids)
partition(X, TimeSeriesSplit(5); time = timestamps)
```

### Single-input shorthand

When `data` is itself the vector a strategy needs (e.g. a timestamps vector for `TimeSplit`, or group IDs for `GroupShuffleSplit`), the corresponding keyword may be omitted — `data` plays both roles. This is governed by the `fallback_from_data` trait.

```julia
partition(timestamps, TimeSplit(); train = 0.7, test = 0.3)
partition(patient_ids, GroupShuffleSplit(); train = 0.8, test = 0.2)
```

## Randomness

Strategies that randomise (e.g. `RandomSplit`, `ShuffleSplit`, `GroupShuffleSplit`) accept an `rng` keyword for reproducibility:

```julia
using Random
partition(X, RandomSplit(); train = 0.8, test = 0.2, rng = MersenneTwister(42))
```

## Materialising splits

`partition` returns indices, not data subsets. Use one of:

- [`splitdata(res, data)`](@ref) — returns a tuple of independent copies via `MLUtils.getobs`.
- [`splitview(res, data)`](@ref) — returns lazy slices via `MLUtils.obsview` (no copy).
- [`trainview`](@ref) / [`testview`](@ref) / [`valview`](@ref) — lazy view of a single cohort across one or more data sources.
- [`traindata`](@ref) / [`testdata`](@ref) / [`valdata`](@ref) — materialised counterparts.

```julia
res = partition(X, KennardStoneSplit(); train = 0.8, test = 0.2)
X_train, X_test = splitdata(res, X)
X_train_view, X_test_view = splitview(res, X)
X_train, y_train = trainview(res, X, y)
```

For a `CrossValidationSplit`, `splitdata` / `splitview` return one element per fold:

```julia
cvs = partition(X, KFold(5))
for (X_tr, X_te) in splitview(cvs, X)
    # train and evaluate
end
```

## Result accessors

Always read indices through the stable accessors rather than the result's fields directly:

- [`trainindices`](@ref) / [`testindices`](@ref) / [`valindices`](@ref)
- [`folds`](@ref) for `CrossValidationSplit`
- [`rowpairs`](@ref) for MLJ-style `(train, test)` tuples
