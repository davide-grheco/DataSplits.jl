using Random
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
    kwval === nothing || throw(SplitInputError("No strategy uses the `$slot=` keyword."))
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


"""
    _resolve_sizes(N, train, validation, test) -> (n_train, n_val, n_test)

Validate and resolve cohort sizes.

**Integer form** — two interpretations, distinguished by the sum:
- sum == `100`: values are percentages of `N`.
- sum == `N`: values are absolute counts.
Any other sum is rejected.

**Float form** — values must each be in `(0, 1)` and sum to approximately 1.0.

When `validation === nothing`, `n_val == 0` and only train and test cohorts
are produced. Rounding remainder is absorbed by `n_test`.
"""
function _resolve_sizes(N::Integer, train::Integer, ::Nothing, test::Integer)
  parts = (:train => train, :test => test)

  _assert_positive_integer(:N, N)
  _assert_positive_integer_parts(parts...)

  n_train, n_test = _resolve_integer_parts(N, parts)

  _check_min_sizes(n_train, n_test)

  return n_train, 0, n_test
end

function _resolve_sizes(N::Integer, train::Integer, validation::Integer, test::Integer)
  parts = (:train => train, :validation => validation, :test => test)

  _assert_positive_integer(:N, N)
  _assert_positive_integer_parts(parts...)

  n_train, n_val, n_test = _resolve_integer_parts(N, parts)

  _check_min_sizes(n_train, n_val, n_test)

  return n_train, n_val, n_test
end

function _resolve_integer_parts(N::Integer, parts::Tuple)
  values = last.(parts)
  s = sum(values)

  if s == 100
    return _resolve_percentage_parts(N, values)
  elseif s == N
    return values
  end

  throw(
    SplitParameterError(
      "Cohort sizes must sum to 100 (percentages) or to N=$N (absolute counts); got $(_format_integer_parts(parts, s)).",
    ),
  )
end

function _resolve_percentage_parts(N::Integer, percentages::Tuple)
  leading = map(p -> round(Int, (p * N) / 100), Base.front(percentages))
  final = N - sum(leading)

  return (leading..., final)
end

function _format_integer_parts(parts::Tuple, total::Integer)
  expr = join(("$(name)($(value))" for (name, value) in parts), " + ")
  return "$expr = $total"
end

function _resolve_sizes(
  N::Integer,
  train::Union{Real,ValidFraction},
  validation::Union{Real,ValidFraction,Nothing},
  test::Union{Real,ValidFraction,Nothing},
)
  _assert_positive_integer(:N, N)
  train = _as_valid_fraction(train)
  validation = _as_valid_fraction(validation)
  test = _as_valid_fraction(test)

  if validation === nothing
    _assert_unit_fraction_sum(train, test)
  else
    _assert_unit_fraction_sum(train, validation, test)
  end

  return _resolve_sizes(N, train, validation, test)
end


function _resolve_sizes(
  N::Integer,
  train::ValidFraction,
  validation::Nothing,
  test::Union{ValidFraction,Nothing},
)
  n_train = round(Int, train * N)
  n_val = 0
  n_test = N - n_train

  _check_min_sizes(n_train, n_test)

  return n_train, n_val, n_test
end

function _resolve_sizes(
  N::Integer,
  train::ValidFraction,
  validation::ValidFraction,
  test::ValidFraction,
)
  n_train = round(Int, train * N)
  n_val = round(Int, validation * N)
  n_test = N - n_train - n_val

  _check_min_sizes(n_train, n_val, n_test)

  return n_train, n_val, n_test
end

function _check_min_sizes(n_train::Integer, n_test::Integer)
  if n_train < 1 || n_test < 1
    throw(
      SplitParameterError(
        "Resolved cohort sizes must each be ≥ 1; got n_train=$n_train, n_test=$n_test.",
      ),
    )
  end

  return nothing
end


function _check_min_sizes(n_train::Integer, n_val::Integer, n_test::Integer)
  if n_train < 1 || n_val < 1 || n_test < 1
    throw(
      SplitParameterError(
        "Resolved cohort sizes must each be ≥ 1; got n_train=$n_train, n_val=$n_val, n_test=$n_test.",
      ),
    )
  end

  return nothing
end


function Base.show(io::IO, res::TrainTestSplit)
  print(
    io,
    "TrainTestSplit  train: $(length(res.train)) obs  test: $(length(res.test)) obs",
  )
end

function Base.show(io::IO, res::TrainValTestSplit)
  print(
    io,
    "TrainValTestSplit  train: $(length(res.train)) obs  val: $(length(res.val)) obs  test: $(length(res.test)) obs",
  )
end

function Base.show(io::IO, res::CrossValidationSplit)
  print(io, "CrossValidationSplit  $(length(res.folds)) folds")
end


# ---------------------------------------------------------------------------
# Result type iteration (enables destructuring: train, test = res)
# ---------------------------------------------------------------------------

Base.iterate(res::TrainTestSplit, state = 1) =
  state == 1 ? (res.train, 2) : state == 2 ? (res.test, 3) : nothing

Base.iterate(res::TrainValTestSplit, state = 1) =
  state == 1 ? (res.train, 2) :
  state == 2 ? (res.val, 3) : state == 3 ? (res.test, 4) : nothing

Base.iterate(res::CrossValidationSplit, state = 1) =
  state <= length(res.folds) ? (res.folds[state], state + 1) : nothing

Base.length(res::TrainTestSplit) = 2
Base.length(res::TrainValTestSplit) = 3
Base.length(res::CrossValidationSplit) = length(res.folds)

"""
    partition(data, alg;
              train, test,
              target=nothing, time=nothing, groups=nothing,
              rng=Random.default_rng()) -> TrainTestSplit

Split `data` into train and test cohorts according to `alg`.

# Cohort sizes (`train`, `test`)

Integers are accepted in two ways:
- **Percentages** — values sum to `100`.
- **Absolute counts** — values sum to `N = numobs(data)`.

Floats in `(0, 1)` summing to `1.0` are also accepted and converted to counts.

# Auxiliary slots

- `target`: response/property vector (e.g. for `SPXYSplit`).
- `time`: temporal ordering vector (e.g. for `TimeSplit`).
- `groups`: group-membership vector (e.g. for `GroupShuffleSplit`).

# Examples
```julia
partition(X, KennardStoneSplit(); train = 80, test = 20)
partition(X, RandomSplit(); train = 0.8, test = 0.2)
partition(X, SPXYSplit(); target = y, train = 80, test = 20)
```
"""
function partition(
  data,
  alg::AbstractSplitStrategy;
  train::Real,
  test::Real,
  target = nothing,
  time = nothing,
  groups = nothing,
  rng = Random.default_rng(),
)
  isempty(data) &&
    throw(SplitInputError("Data must not be empty. Please provide a non-empty dataset."))
  N = numobs(data)
  N >= 2 || throw(SplitInputError("Cannot split fewer than 2 observations."))

  n_train, _, n_test = _resolve_sizes(N, train, nothing, test)

  algs = (alg,)
  resolved_target = _resolve_slot(algs, data, target, :target)
  resolved_time = _resolve_slot(algs, data, time, :time)
  resolved_groups = _resolve_slot(algs, data, groups, :groups)

  data_internal = :data ∈ consumes(alg) ? _to_feature_matrix(data) : data

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

"""
    partition(data, alg, val_alg;
              train, validation, test,
              target=nothing, time=nothing, groups=nothing,
              rng=Random.default_rng()) -> TrainValTestSplit

Split `data` into train, validation, and test cohorts using two strategies.

`alg` separates the test cohort from the rest; `val_alg` then separates the
validation cohort from the remaining train pool.

# Cohort sizes (`train`, `validation`, `test`)

Integers are accepted in two ways:
- **Percentages** — values sum to `100`.
- **Absolute counts** — values sum to `N = numobs(data)`.

Floats in `(0, 1)` summing to `1.0` are also accepted.

# Examples
```julia
partition(X, RandomSplit(), KennardStoneSplit();
          train = 70, validation = 10, test = 20)
partition(X, RandomSplit(), KennardStoneSplit();
          train = 0.7, validation = 0.1, test = 0.2)
```
"""
function partition(
  data,
  alg::AbstractSplitStrategy,
  val_alg::AbstractSplitStrategy;
  train::Real,
  validation::Real,
  test::Real,
  target = nothing,
  time = nothing,
  groups = nothing,
  rng = Random.default_rng(),
)
  isempty(data) &&
    throw(SplitInputError("Data must not be empty. Please provide a non-empty dataset."))
  N = numobs(data)
  N >= 2 || throw(SplitInputError("Cannot split fewer than 2 observations."))

  n_train, n_val, n_test = _resolve_sizes(N, train, validation, test)

  algs = (alg, val_alg)
  resolved_target = _resolve_slot(algs, data, target, :target)
  resolved_time = _resolve_slot(algs, data, time, :time)
  resolved_groups = _resolve_slot(algs, data, groups, :groups)

  needs_matrix = any(a -> :data ∈ consumes(a), algs)
  data_internal = needs_matrix ? _to_feature_matrix(data) : data

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
  inner_target = resolved_target === nothing ? nothing : view(resolved_target, train_pool)
  inner_time = resolved_time === nothing ? nothing : view(resolved_time, train_pool)
  inner_groups = resolved_groups === nothing ? nothing : view(resolved_groups, train_pool)

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
