```@meta
CurrentModule = DataSplits
```

# Core API Reference

DataSplits expects matrices to follow the Julia ML convention: **columns are samples, rows are features**. Transpose row-major data before passing it in. For Tables.jl inputs rows are samples — the conversion to F×N is handled internally where required.

All split strategies return indices in `1:N`. Materialise subsets through [`splitdata`](@ref) / [`splitview`](@ref) (or the cohort-specific accessors below) rather than indexing directly.

## `partition`

```julia
partition(data, alg::AbstractSplitStrategy;
          train, test,
          target=nothing, time=nothing, groups=nothing,
          rng=Random.default_rng()) -> TrainTestSplit

partition(data, test_alg, val_alg;
          train, validation, test,
          target=nothing, time=nothing, groups=nothing,
          rng=Random.default_rng()) -> TrainValTestSplit

partition(data, alg::AbstractCVStrategy;
          target=nothing, time=nothing, groups=nothing,
          rng=Random.default_rng()) -> CrossValidationSplit

partition(data, alg::AbstractResamplingCVStrategy;
          train, test,
          target=nothing, time=nothing, groups=nothing,
          rng=Random.default_rng()) -> CrossValidationSplit
```

The 2-cohort form returns a [`TrainTestSplit`](@ref). The 3-cohort form uses `test_alg` to separate the test cohort from the rest, then `val_alg` to separate the validation cohort from the remaining train pool, and returns a [`TrainValTestSplit`](@ref). The CV forms return a [`CrossValidationSplit`](@ref) wrapping one fold per element.

## Strategy types

```julia
abstract type AbstractSplitStrategy end
abstract type AbstractCVStrategy <: AbstractSplitStrategy end
abstract type AbstractResamplingCVStrategy <: AbstractCVStrategy end
```

Dispatch on these governs the `partition` method called:

- [`AbstractSplitStrategy`](@ref) — single-pass strategies, called with `train`/`test` (or `train`/`validation`/`test` in the 3-cohort form).
- [`AbstractCVStrategy`](@ref) — CV strategies whose fold sizes are fixed by the strategy itself (`k`, `n_splits`). Called without `train`/`test`.
- [`AbstractResamplingCVStrategy`](@ref) — CV strategies whose folds are independent random splits sized by the caller. Called with `train`/`test`.

## Result types

```julia
struct TrainTestSplit{I} <: AbstractSplitResult
    train::I
    test::I
end

struct TrainValTestSplit{I} <: AbstractSplitResult
    train::I
    val::I
    test::I
end

struct CrossValidationSplit{T<:AbstractSplitResult} <: AbstractSplitResult
    folds::Vector{T}
end
```

All three iterate naturally — `train, test = res`, `train, val, test = res3`, `for fold in cvs`. `CrossValidationSplit` additionally supports `cvs[i]`, `cvs[range]`, `first(cvs)`, `last(cvs)`, `length(cvs)`.

Strategies should never expose these fields directly — read indices via the accessors below.

## Result accessors

```julia
trainindices(res) -> indices
testindices(res)  -> indices
valindices(res::TrainValTestSplit) -> indices
folds(res::CrossValidationSplit)   -> Vector{<:AbstractSplitResult}
rowpairs(res)                       -> Vector{Tuple{Vector{Int}, Vector{Int}}}
```

[`rowpairs`](@ref) returns the `(train, test)` tuple format accepted by MLJ's `evaluate!` `resampling=` keyword.

## Materialising splits

[`splitdata`](@ref) and [`splitview`](@ref) take a result and the original `data`, returning the corresponding subsets:

```julia
splitdata(res::TrainTestSplit, data)      # (X_train, X_test) via getobs
splitview(res::TrainTestSplit, data)      # (X_train, X_test) via obsview (lazy)

splitdata(res::TrainValTestSplit, data)   # (X_train, X_val, X_test)
splitdata(res::CrossValidationSplit, data) # Vector of per-fold tuples
```

For per-cohort access, possibly over multiple data sources at once:

```julia
trainview(res, data...)  # lazy
testview(res, data...)
valview(res, data...)    # TrainValTestSplit only

traindata(res, data...)  # materialised via getobs
testdata(res, data...)
valdata(res, data...)
```

When called with a single data source the result is the view (or copy) directly; with two or more it is a `Tuple` — convenient for `Flux.DataLoader` and similar.

## Trait surface

Strategy authors declare which auxiliary slots a strategy reads, and which of those can fall back to `data` itself when the corresponding keyword is omitted:

```julia
consumes(alg)            -> NTuple{N,Symbol}   # subset of (:data, :target, :time, :groups)
fallback_from_data(alg)  -> NTuple{N,Symbol}   # subset of consumes(alg)
```

Examples:

```julia
consumes(::SPXYSplit)          = (:data, :target)
fallback_from_data(::SPXYSplit) = ()           # target must be supplied

consumes(::TimeSplit)          = (:time,)
fallback_from_data(::TimeSplit) = (:time,)     # data can be the time vector itself
```

`fallback_from_data(alg) ⊆ consumes(alg)` is required. See [Extending DataSplits](80-extending.md) for how these traits feed into `partition`'s slot resolution.

## Exceptions

```julia
SplitInputError           # malformed inputs (wrong shape, missing required slot)
SplitParameterError       # invalid parameters (cohort sizes, k, fraction out of range)
SplitNotImplementedError  # a method (e.g. splitdata) is missing for a custom result type
```
