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
- `_partition(data, alg; n_train, n_test, target, time, groups, rng)`
  returning an `AbstractSplitResult`.
- `consumes(::MyStrategy)` returning a tuple of symbols from
  `(:data, :target, :time, :groups)`.
- `fallback_from_data(::MyStrategy)` returning the subset of `consumes`
  that can fall back to `data`.
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
res = partition(X, KennardStoneSplit(); train = 80, test = 20)
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
res = partition(X, RandomSplit(), KennardStoneSplit();
                train = 70, validation = 10, test = 20)
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
"""
consumes(::AbstractSplitStrategy) = ()

"""
    fallback_from_data(alg::AbstractSplitStrategy) -> NTuple{N, Symbol}

Return the subset of `consumes(alg)` whose keyword may be omitted in
`partition`, in which case `data` itself fills that slot.

Must satisfy: `fallback_from_data(alg) ⊆ consumes(alg)`.
"""
fallback_from_data(::AbstractSplitStrategy) = ()


# ---------------------------------------------------------------------------
# Tables.jl helper
# ---------------------------------------------------------------------------

# Convert a Tables.jl-compatible input (e.g. DataFrame) to a features×samples
# matrix (F×N), which is the internal convention for distance-based strategies.
# Non-table inputs are returned unchanged.
function _to_feature_matrix(X)
  Tables.istable(X) ? permutedims(Tables.matrix(X)) : X
end


# ---------------------------------------------------------------------------
# Slot resolution (works for one or two strategies via a tuple of algs)
# ---------------------------------------------------------------------------

function _resolve_slot(algs::Tuple, data, kwval, slot::Symbol)
  consumers = filter(a -> slot ∈ consumes(a), algs)
  if isempty(consumers)
    kwval === nothing ||
      throw(SplitInputError("No strategy uses the `$slot=` keyword."))
    return nothing
  end
  if kwval === nothing
    if all(a -> slot ∈ fallback_from_data(a), consumers)
      kwval = data
    else
      bad = first(filter(a -> slot ∉ fallback_from_data(a), consumers))
      throw(SplitInputError("$(typeof(bad)) requires the `$slot=` keyword."))
    end
  end
  kwval isa AbstractVector ||
    throw(SplitInputError("`$slot` must be a 1D AbstractVector, got $(typeof(kwval))."))
  return kwval
end

# For the per-strategy call: only forward the slot if this strategy consumes it.
_slot_for(alg, value, slot::Symbol) = slot ∈ consumes(alg) ? value : nothing


# ---------------------------------------------------------------------------
# Cohort size resolution
# ---------------------------------------------------------------------------

"""
    _resolve_sizes(N, train, validation, test) -> (n_train, n_val, n_test)

Validate and resolve cohort sizes from integer keywords.

Two interpretations are accepted, distinguished by the sum of the values:

- if they sum to **100**, they are treated as **percentages of `N`**;
- if they sum to **`N`**, they are treated as **absolute counts**.

Any other sum is rejected. When `validation === nothing`, the result has
`n_val == 0` and only train and test cohorts are produced.

When percentages do not divide `N` evenly, `n_train` and `n_val` are
rounded to the nearest integer and `n_test` absorbs the remainder so that
`n_train + n_val + n_test == N`.
"""
function _resolve_sizes(
  N::Integer,
  train::Integer,
  validation::Union{Integer,Nothing},
  test::Integer,
)
  three_cohort = validation !== nothing

  for (k, v) in (
    three_cohort ?
    ((:train, train), (:validation, validation), (:test, test)) :
    ((:train, train), (:test, test))
  )
    v >= 1 ||
      throw(SplitParameterError("`$k` must be a positive integer, got $v."))
  end

  s = three_cohort ? train + validation + test : train + test

  if s == 100
    n_train = round(Int, (train * N) / 100)
    n_val = three_cohort ? round(Int, (validation * N) / 100) : 0
    n_test = N - n_train - n_val
  elseif s == N
    n_train = train
    n_val = three_cohort ? validation : 0
    n_test = test
  else
    parts = three_cohort ?
      "train($train) + validation($validation) + test($test) = $s" :
      "train($train) + test($test) = $s"
    throw(
      SplitParameterError(
        "Cohort sizes must sum to 100 (percentages) or to N=$N (absolute counts); got $parts.",
      ),
    )
  end

  if n_train < 1 || n_test < 1 || (three_cohort && n_val < 1)
    throw(
      SplitParameterError(
        "Resolved cohort sizes must each be ≥ 1; got n_train=$n_train, n_val=$n_val, n_test=$n_test.",
      ),
    )
  end

  return n_train, n_val, n_test
end


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

"""
    partition(data, alg [, val_alg];
              train, test, validation=nothing,
              target=nothing, time=nothing, groups=nothing,
              rng=Random.default_rng()) -> AbstractSplitResult

Split `data` into train/test (or train/validation/test) according to one or
two splitting strategies.

# Arguments
- `data`: Observation container (matrix, vector, DataFrame, …).
  Columns are samples for matrices; rows are samples for DataFrames.
- `alg`: A splitting strategy. With a single strategy `alg`, this strategy
  separates train from test. With two strategies, `alg` separates the test
  cohort from the rest.
- `val_alg` *(optional)*: When given, this second strategy separates the
  validation cohort from the train pool produced by `alg`.

# Cohort sizes (`train`, `validation`, `test`)

All sizes are positive integers. Two interpretations are accepted:

- **Percentages** — values sum to `100`.
- **Absolute counts** — values sum to `N = numobs(data)`.

Any other sum is rejected.

# Auxiliary slots

- `target`: response/property vector used by some strategies (e.g. `SPXYSplit`).
- `time`: temporal ordering vector used by `TimeSplit` variants.
- `groups`: group-membership vector used by `GroupShuffleSplit` /
  `GroupStratifiedSplit`.

When a strategy's required slot is omitted and that slot is in
`fallback_from_data(alg)`, `data` itself fills the role
(e.g. `partition(dates, TimeSplitOldest(); train=70, test=30)`).

# Examples
```julia
# 2 cohorts, percentages
partition(X, KennardStoneSplit(); train = 80, test = 20)

# 2 cohorts, absolute counts (N = 250)
partition(X, RandomSplit(); train = 200, test = 50)

# 3 cohorts, mixed strategies
partition(X, RandomSplit(), KennardStoneSplit();
          train = 70, validation = 10, test = 20)

# Slot keywords still apply
partition(X, SPXYSplit(); target = y, train = 80, test = 20)
```
"""
function partition(
  data,
  alg::AbstractSplitStrategy,
  val_alg::Union{Nothing,AbstractSplitStrategy} = nothing;
  train::Integer,
  test::Integer,
  validation::Union{Integer,Nothing} = nothing,
  target = nothing,
  time = nothing,
  groups = nothing,
  rng = Random.default_rng(),
)
  isempty(data) &&
    throw(SplitInputError("Data must not be empty. Please provide a non-empty dataset."))
  N = numobs(data)
  N >= 2 || throw(SplitInputError("Cannot split fewer than 2 observations."))

  if val_alg === nothing && validation !== nothing
    throw(SplitInputError("`validation=` requires a second positional strategy."))
  end
  if val_alg !== nothing && validation === nothing
    throw(SplitInputError("Two strategies require the `validation=` keyword."))
  end

  n_train, n_val, n_test = _resolve_sizes(N, train, validation, test)

  algs = val_alg === nothing ? (alg,) : (alg, val_alg)
  resolved_target = _resolve_slot(algs, data, target, :target)
  resolved_time = _resolve_slot(algs, data, time, :time)
  resolved_groups = _resolve_slot(algs, data, groups, :groups)

  needs_matrix = any(a -> :data ∈ consumes(a), algs)
  data_internal = needs_matrix ? _to_feature_matrix(data) : data

  if val_alg === nothing
    return _partition(
      data_internal,
      alg;
      n_train = n_train,
      n_test = n_test,
      target = _slot_for(alg, resolved_target, :target),
      time = _slot_for(alg, resolved_time, :time),
      groups = _slot_for(alg, resolved_groups, :groups),
      rng = rng,
    )
  end

  outer = _partition(
    data_internal,
    alg;
    n_train = n_train + n_val,
    n_test = n_test,
    target = _slot_for(alg, resolved_target, :target),
    time = _slot_for(alg, resolved_time, :time),
    groups = _slot_for(alg, resolved_groups, :groups),
    rng = rng,
  )
  outer isa TrainTestSplit || throw(
    SplitNotImplementedError(
      "First strategy must return a TrainTestSplit, got $(typeof(outer)).",
    ),
  )

  train_pool = outer.train
  test_idx = outer.test

  data_inner = obsview(data_internal, train_pool)
  inner_target =
    resolved_target === nothing ? nothing : view(resolved_target, train_pool)
  inner_time = resolved_time === nothing ? nothing : view(resolved_time, train_pool)
  inner_groups =
    resolved_groups === nothing ? nothing : view(resolved_groups, train_pool)

  inner = _partition(
    data_inner,
    val_alg;
    n_train = n_train,
    n_test = n_val,
    target = _slot_for(val_alg, inner_target, :target),
    time = _slot_for(val_alg, inner_time, :time),
    groups = _slot_for(val_alg, inner_groups, :groups),
    rng = rng,
  )
  inner isa TrainTestSplit || throw(
    SplitNotImplementedError(
      "Validation strategy must return a TrainTestSplit, got $(typeof(inner)).",
    ),
  )

  return TrainValTestSplit(train_pool[inner.train], train_pool[inner.test], test_idx)
end
