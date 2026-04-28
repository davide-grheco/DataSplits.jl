using Random
using ArrayInterface
using MLUtils: numobs, getobs, obsview
using Tables


# ---------------------------------------------------------------------------
# Abstract types
# ---------------------------------------------------------------------------

"""
    AbstractSplitStrategy

Abstract supertype for all splitting strategies.

To implement a custom strategy, subtype this and define:
- `_partition(data, alg; target, time, groups, rng)` returning an `AbstractSplitResult`
- `consumes(::MyStrategy)` returning a tuple of symbols from `(:data, :target, :time, :groups)`
- `fallback_from_data(::MyStrategy)` returning the subset of `consumes` that can fall back to `data`
"""
abstract type AbstractSplitStrategy end

"""
    AbstractSplitResult

Abstract supertype for all split result types.
"""
abstract type AbstractSplitResult end


# ---------------------------------------------------------------------------
# Concrete result types
# ---------------------------------------------------------------------------

"""
    TrainTestSplit

A result type representing a train/test split.

# Fields
- `train`: Indices of training samples.
- `test`: Indices of test samples.

# Examples
```julia
res = partition(X, KennardStoneSplit(0.8))
X_train, X_test = splitdata(res, X)
```
"""
struct TrainTestSplit{I} <: AbstractSplitResult
  train::I
  test::I
end

"""
    TrainValTestSplit

A result type representing a train/validation/test split.

# Fields
- `train`: Indices of training samples.
- `val`: Indices of validation samples.
- `test`: Indices of test samples.

# Examples
```julia
res = partition(X, SomeTrainValTestStrategy(...))
X_train, X_val, X_test = splitdata(res, X)
```
"""
struct TrainValTestSplit{I} <: AbstractSplitResult
  train::I
  val::I
  test::I
end

"""
    CrossValidationSplit

A result type representing a k-fold cross-validation split.

# Fields
- `folds::Vector{<:AbstractSplitResult}`: One result per fold.

# Examples
```julia
res = partition(X, SomeCVStrategy(...))
for (X_train, X_test) in splitdata(res, X)
    # ...
end
```
"""
struct CrossValidationSplit{T<:AbstractSplitResult} <: AbstractSplitResult
  folds::Vector{T}
end


# ---------------------------------------------------------------------------
# Result accessors (stable contract — do not access fields directly)
# ---------------------------------------------------------------------------

"""
    trainindices(res::AbstractSplitResult) -> indices

Return the training indices from a split result.
"""
trainindices(res::TrainTestSplit) = res.train
trainindices(res::TrainValTestSplit) = res.train

"""
    testindices(res::AbstractSplitResult) -> indices

Return the test indices from a split result.
"""
testindices(res::TrainTestSplit) = res.test
testindices(res::TrainValTestSplit) = res.test

"""
    valindices(res::TrainValTestSplit) -> indices

Return the validation indices from a split result.
"""
valindices(res::TrainValTestSplit) = res.val

"""
    folds(res::CrossValidationSplit) -> Vector{<:AbstractSplitResult}

Return the individual fold results from a cross-validation split.
"""
folds(res::CrossValidationSplit) = res.folds


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

"""
    splitdata(res, data)

Materialise the split: return a tuple of data subsets corresponding to the
train/test (and optionally validation) indices in `res`.

When `data` is a DataFrame or other Tables.jl-compatible container,
`splitdata` returns subsets of the same type.

# Examples
```julia
res = partition(X, KennardStoneSplit(0.8))
X_train, X_test = splitdata(res, X)
```
"""
function splitdata(res::AbstractSplitResult, data)
  throw(
    SplitNotImplementedError(
      "splitdata is not implemented for $(typeof(res)). Implement splitdata(::$(typeof(res)), data).",
    ),
  )
end

splitdata(res::TrainTestSplit, data) = (getobs(data, res.train), getobs(data, res.test))

splitdata(res::TrainValTestSplit, data) =
  (getobs(data, res.train), getobs(data, res.val), getobs(data, res.test))

splitdata(res::CrossValidationSplit, data) = [splitdata(fold, data) for fold in res.folds]

"""
    splitview(res, data)

Like `splitdata` but returns lazy views via `MLUtils.obsview` — no data is
copied. Prefer `splitdata` when you need independent copies.

# Examples
```julia
res = partition(X, RandomSplit(0.8))
X_train, X_test = splitview(res, X)
```
"""
splitview(res::TrainTestSplit, data) = (obsview(data, res.train), obsview(data, res.test))

splitview(res::TrainValTestSplit, data) =
  (obsview(data, res.train), obsview(data, res.val), obsview(data, res.test))

splitview(res::CrossValidationSplit, data) = [splitview(fold, data) for fold in res.folds]


# ---------------------------------------------------------------------------
# Trait system
# ---------------------------------------------------------------------------

"""
    consumes(alg::AbstractSplitStrategy) -> NTuple{N, Symbol}

Return the named slots this strategy reads, as a tuple of symbols from
`(:data, :target, :time, :groups)`.

- `:data` — the strategy inspects observation values (e.g. for distances).
  Omitting it means only `numobs` is needed (e.g. `RandomSplit`).
- `:target` — the strategy reads a response/property vector.
- `:time` — the strategy reads a temporal ordering vector.
- `:groups` — the strategy reads a group-membership vector.
"""
consumes(::AbstractSplitStrategy) = ()

"""
    fallback_from_data(alg::AbstractSplitStrategy) -> NTuple{N, Symbol}

Return the subset of `consumes(alg)` whose keyword may be omitted in
`partition`, in which case `data` itself fills that slot.

Must satisfy: `fallback_from_data(alg) ⊆ consumes(alg)`.
"""
fallback_from_data(::AbstractSplitStrategy) = ()

"""
Convert a Tables.jl-compatible input (e.g. DataFrame) to a features×samples
matrix (F×N), which is the internal convention for distance-based strategies.
Non-table inputs are returned unchanged.
"""
function _to_feature_matrix(X)
  Tables.istable(X) ? transpose(Tables.matrix(X)) : X
end


# ---------------------------------------------------------------------------
# Auxiliary slot resolution
# ---------------------------------------------------------------------------

function _resolve_slot(alg, data, kwval, slot::Symbol)
  if slot ∈ consumes(alg)
    if kwval === nothing
      if slot ∈ fallback_from_data(alg)
        kwval = data
      else
        throw(SplitInputError("$(typeof(alg)) requires the `$slot=` keyword."))
      end
    end
    kwval isa AbstractVector ||
      throw(SplitInputError("`$slot` must be a 1D AbstractVector, got $(typeof(kwval))."))
    return kwval
  else
    kwval === nothing ||
      throw(SplitInputError("$(typeof(alg)) does not use the `$slot=` keyword."))
    return nothing
  end
end


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

"""
    partition(data, alg::AbstractSplitStrategy;
              target=nothing, time=nothing, groups=nothing,
              rng=Random.default_rng()) -> AbstractSplitResult

Split `data` into train/test (or train/val/test, or cross-validation folds)
according to `alg`.

# Arguments
- `data`: Observation container (matrix, vector, DataFrame, …).
  Columns are samples for matrices; rows are samples for DataFrames.
- `alg`: A splitting strategy.

# Keywords
- `target`: Response / property vector used by some strategies (e.g. `SPXYSplit`).
- `time`: Temporal ordering vector used by `TimeSplit` variants.
- `groups`: Group-membership vector used by `GroupShuffleSplit` / `GroupStratifiedSplit`.
- `rng`: Random number generator.

# Fallback rule
When a strategy's required keyword is omitted and `fallback_from_data(alg)`
includes that slot, `data` itself is used for that role. This makes
single-input calls natural:

```julia
partition(dates, TimeSplitOldest(0.8))          # dates is both data and time
partition(y, TargetPropertyHigh(0.8))           # y is both data and target
partition(ids, GroupShuffleSplit(0.8))          # ids is both data and groups
```

# Examples
```julia
res = partition(X, KennardStoneSplit(0.8))
res = partition(X, SPXYSplit(0.7); target=y)
res = partition(X, GroupShuffleSplit(0.8); groups=patient_ids)
df_train, df_test = splitdata(res, df)
```
"""
function partition(
  data,
  alg::AbstractSplitStrategy;
  target = nothing,
  time = nothing,
  groups = nothing,
  rng = Random.default_rng(),
)
  isempty(data) &&
    throw(SplitInputError("Data must not be empty. Please provide a non-empty dataset."))
  numobs(data) < 2 && throw(SplitInputError("Cannot split fewer than 2 observations."))

  resolved_target = _resolve_slot(alg, data, target, :target)
  resolved_time = _resolve_slot(alg, data, time, :time)
  resolved_groups = _resolve_slot(alg, data, groups, :groups)

  data_internal = :data ∈ consumes(alg) ? _to_feature_matrix(data) : data

  return _partition(
    data_internal,
    alg;
    target = resolved_target,
    time = resolved_time,
    groups = resolved_groups,
    rng = rng,
  )
end


# ---------------------------------------------------------------------------
# ValidFraction
# ---------------------------------------------------------------------------

struct ValidFraction{T<:Real}
  frac::T
  function ValidFraction(frac::T) where {T<:Real}
    if !(0 < frac < 1)
      throw(SplitParameterError("Fraction must be strictly between 0 and 1. Got $frac."))
    end
    new{T}(frac)
  end
end

Base.:*(vf::ValidFraction, x::Number) = vf.frac * x
Base.:*(x::Number, vf::ValidFraction) = x * vf.frac
Base.:+(x::Number, vf::ValidFraction) = x + vf.frac
Base.:-(x::Number, vf::ValidFraction) = x - vf.frac
Base.float(vf::ValidFraction) = vf.frac
Base.convert(::Type{T}, vf::ValidFraction) where {T<:Number} = convert(T, vf.frac)
