```@meta
CurrentModule = DataSplits
```

# Extending DataSplits

DataSplits is built around a small extension protocol: subtype an abstract strategy type, declare which auxiliary slots you consume, and implement a single `_partition` method. `partition` handles cohort-size resolution, slot validation, and result wrapping.

## Checklist

1. Choose the right abstract supertype.
   - [`AbstractSplitStrategy`](@ref) — single-pass strategies that return a [`TrainTestSplit`](@ref) (or [`TrainValTestSplit`](@ref)).
   - [`AbstractCVStrategy`](@ref) — cross-validation strategies whose fold sizes are fixed by the strategy itself (typically via `k`).
   - [`AbstractResamplingCVStrategy`](@ref) — CV strategies whose folds are independent random splits sized by the caller (`train`/`test` per fold).
2. Declare the auxiliary slots your strategy reads with [`consumes`](@ref) and (optionally) [`fallback_from_data`](@ref).
3. Implement [`_partition`](@ref) with the signature matching your supertype.
4. Add a test file under `test/test-<name>.jl` — `test/runtests.jl` discovers any `test-*.jl` automatically.
5. Update `src/DataSplits.jl` to `include` the new file and export the strategy.

## Traits

```julia
consumes(alg)           -> NTuple{N,Symbol}    # subset of (:data, :target, :time, :groups)
fallback_from_data(alg) -> NTuple{N,Symbol}    # subset of consumes(alg)
```

`consumes` lists every slot the strategy reads — `partition` uses this to validate keyword arguments and length-check the slot vectors. `fallback_from_data` is the subset of consumed slots for which `data` itself may stand in when the caller omits the keyword (e.g. `partition(timestamps, TimeSplit(); …)` works because `TimeSplit` declares `fallback_from_data = (:time,)`).

The default is `()` for both — your strategy reads no auxiliary slots and accepts no fallbacks.

## `_partition` contract

```julia
# AbstractSplitStrategy — single-pass
_partition(data, alg::MyStrategy;
           n_train, n_test,
           target, time, groups,
           rng,
           kwargs...) -> AbstractSplitResult

# AbstractCVStrategy — fold sizes intrinsic to the strategy
_partition(data, alg::MyCV;
           target, time, groups,
           rng,
           kwargs...) -> CrossValidationSplit

# AbstractResamplingCVStrategy — caller-sized resamples
_partition(data, alg::MyResamplingCV;
           n_train, n_test,
           target, time, groups,
           rng,
           kwargs...) -> CrossValidationSplit
```

The arguments are post-resolution: `n_train` and `n_test` are integer counts; `target`, `time`, and `groups` are either the resolved vectors or `nothing` (only forwarded if your strategy declares them in `consumes`). Always accept `kwargs...` so the signature stays forward-compatible as new slots are added.

Use [`MLUtils.numobs`](https://juliaml.github.io/MLUtils.jl/) for the observation count and [`MLUtils.getobs`](https://juliaml.github.io/MLUtils.jl/) / [`MLUtils.obsview`](https://juliaml.github.io/MLUtils.jl/) for container-agnostic access. Return indices in `1:N` — never the materialised data.

## Example: a single-pass strategy

A strategy that puts the first `n_train` observations (in container order) into the train cohort:

```julia
using DataSplits, MLUtils

struct FirstNSplit <: AbstractSplitStrategy end

consumes(::FirstNSplit) = ()
fallback_from_data(::FirstNSplit) = ()

function DataSplits._partition(data, ::FirstNSplit; n_train, n_test, kwargs...)
    N = numobs(data)
    return TrainTestSplit(1:n_train, (n_train + 1):N)
end

res = partition(X, FirstNSplit(); train = 0.8, test = 0.2)
X_train, X_test = splitdata(res, X)
```

## Example: a CV strategy consuming groups

A toy `OddEvenGroupCV` that puts odd group IDs in test for fold 1 and even ones in test for fold 2:

```julia
using DataSplits, MLUtils

struct OddEvenGroupCV <: AbstractCVStrategy end

consumes(::OddEvenGroupCV) = (:groups,)
fallback_from_data(::OddEvenGroupCV) = (:groups,)

function DataSplits._partition(data, ::OddEvenGroupCV; groups, kwargs...)
    N = numobs(data)
    odd_idx  = findall(g -> isodd(g),  groups)
    even_idx = findall(g -> iseven(g), groups)
    f1 = TrainTestSplit(even_idx, odd_idx)
    f2 = TrainTestSplit(odd_idx,  even_idx)
    return CrossValidationSplit([f1, f2])
end

cvs = partition(X, OddEvenGroupCV(); groups = group_ids)
```

Because `(:groups,)` is also in `fallback_from_data`, the shorthand `partition(group_ids, OddEvenGroupCV())` works too.

## Custom result types

If your strategy returns a custom subtype of [`AbstractSplitResult`](@ref) rather than `TrainTestSplit` / `TrainValTestSplit` / `CrossValidationSplit`, also implement:

- [`splitdata`](@ref) and [`splitview`](@ref) — for the two-argument materialisation calls.
- [`trainindices`](@ref), [`testindices`](@ref), and (if applicable) [`valindices`](@ref) — the stable accessor contract.
- Iteration if you want destructuring (`train, test = res`).

See `NestedFold` in `src/strategies/NestedCV.jl` for a worked example of a custom result type with all of the above.

## Validation

Use `ValidFraction` only if your *strategy struct* takes a fraction as a constructor argument (e.g. an internal swap rate); cohort-size validation is handled by `partition` and does not need to be repeated. Throw `SplitParameterError` for invalid parameters and `SplitInputError` for malformed inputs — see [Core API Reference](03-core-api-reference.md#Exceptions).
