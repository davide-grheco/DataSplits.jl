using Random
using MLUtils: numobs, obsview

"""
    NestedFold{I} <: AbstractSplitResult

A single outer fold of a nested cross-validation split.

# Fields
- `train::I` — outer training indices (absolute, into `1:N`).
- `test::I` — outer test indices (absolute, into `1:N`).
- `inner::CrossValidationSplit{TrainTestSplit{I}}` — the inner cross-validation
  produced by applying the inner strategy to the outer training cohort. Inner
  fold indices are **absolute** (already remapped from the local
  `1:length(train)` index space back into `1:N`), so they can be used
  directly against the original `data`.

# Iteration
`NestedFold` iterates as `(train, test)` — identical to `TrainTestSplit`
for outer-loop compatibility. To access the inner CV, use the
[`innerfolds`](@ref) accessor or the `inner` field.
"""
struct NestedFold{I} <: AbstractSplitResult
  train::I
  test::I
  inner::CrossValidationSplit{TrainTestSplit{I}}
end

trainindices(f::NestedFold) = f.train
testindices(f::NestedFold) = f.test

"""
    innerfolds(f::NestedFold) -> CrossValidationSplit

Return the inner cross-validation split associated with the outer fold `f`.
Inner fold indices are absolute (into `1:N`), so they can be used directly
against the original `data` without further remapping.

# Example
```julia
cvs = partition(X, NestedCV(KFold(5), KFold(3)))
for outerfold in folds(cvs)
    X_outer_train, X_outer_test = splitdata(outerfold, X)
    for (X_tr, X_val) in splitview(innerfolds(outerfold), X)
        # hyperparameter tuning on the outer training cohort
    end
end
```
"""
innerfolds(f::NestedFold) = f.inner

splitdata(res::NestedFold, data) = (getobs(data, res.train), getobs(data, res.test))
splitview(res::NestedFold, data) = (obsview(data, res.train), obsview(data, res.test))

trainview(r::NestedFold, data...) = _co_views(r.train, data...)
testview(r::NestedFold, data...) = _co_views(r.test, data...)
traindata(r::NestedFold, data...) = _co_data(r.train, data...)
testdata(r::NestedFold, data...) = _co_data(r.test, data...)

Base.iterate(res::NestedFold, state = 1) =
  state == 1 ? (res.train, 2) : state == 2 ? (res.test, 3) : nothing
Base.length(::NestedFold) = 2

function Base.show(io::IO, res::NestedFold)
  print(
    io,
    "NestedFold  train: $(length(res.train)) obs  test: $(length(res.test)) obs  inner: $(length(folds(res.inner))) folds",
  )
end


"""
    NestedCV(outer::AbstractCVStrategy, inner::AbstractCVStrategy) <: AbstractCVStrategy

Nested cross-validation — combine an outer CV (for unbiased performance
estimation) with an inner CV (for hyperparameter selection within each
outer training cohort).

For each fold of `outer`, the outer test cohort is held out and the
outer training cohort is itself partitioned by `inner`. The result is a
`CrossValidationSplit{NestedFold}` where each fold exposes the usual
outer `(train, test)` pair plus an `innerfolds(...)` accessor giving
the inner CV split. Inner fold indices are already remapped to the
absolute `1:N` index space.

# Fields
- `outer::AbstractCVStrategy` — produces the outer train/test split per fold.
- `inner::AbstractCVStrategy` — partitions each outer training cohort.

# Restrictions
- `outer` must produce `CrossValidationSplit{TrainTestSplit}` (i.e. not nested
  itself).
- `inner` must be a non-resampling [`AbstractCVStrategy`](@ref) — strategies
  subtyping [`AbstractResamplingCVStrategy`](@ref) (e.g. `ShuffleSplit`,
  `StratifiedShuffleSplit`, `GroupShuffleSplitCV`) require
  caller-set cohort sizes which `NestedCV` does not currently propagate.

# Slot resolution
`consumes(::NestedCV)` is the union of the outer and inner strategies'
declared slots. `partition` resolves each slot once against the full
dataset, then the inner CV sees a view sliced to the outer training cohort.

# Examples
```julia
# 5 × 3 nested k-fold on a classification target.
cvs = partition(X, NestedCV(StratifiedKFold(5), StratifiedKFold(3)); target = y)
for outerfold in folds(cvs)
    X_tr_outer, X_te_outer = splitdata(outerfold, X)
    y_tr_outer, _          = splitdata(outerfold, y)
    for (X_tr, X_val) in splitview(innerfolds(outerfold), X)
        # tune hyperparameters on (X_tr, X_val)
    end
    # then refit on the full outer training cohort and score on X_te_outer
end

# Group-aware nesting.
cvs = partition(X, NestedCV(GroupKFold(5), GroupKFold(3)); groups = patient_ids)
```
"""
struct NestedCV{O<:AbstractCVStrategy,I<:AbstractCVStrategy} <: AbstractCVStrategy
  outer::O
  inner::I
end

function NestedCV(outer::AbstractCVStrategy, inner::AbstractCVStrategy)
  inner isa AbstractResamplingCVStrategy && throw(
    SplitParameterError(
      "NestedCV: inner strategy must be a non-resampling AbstractCVStrategy (got $(typeof(inner))). " *
      "Resampling strategies require caller-set cohort sizes which NestedCV does not propagate.",
    ),
  )
  NestedCV{typeof(outer),typeof(inner)}(outer, inner)
end

function consumes(nc::NestedCV)
  seen = Symbol[]
  for s in consumes(nc.outer)
    s in seen || push!(seen, s)
  end
  for s in consumes(nc.inner)
    s in seen || push!(seen, s)
  end
  return Tuple(seen)
end

function fallback_from_data(nc::NestedCV)
  outer_consumes = consumes(nc.outer)
  inner_consumes = consumes(nc.inner)
  outer_fb = fallback_from_data(nc.outer)
  inner_fb = fallback_from_data(nc.inner)
  result = Symbol[]
  for s in consumes(nc)
    used_by_outer = s in outer_consumes
    used_by_inner = s in inner_consumes
    ok_outer = !used_by_outer || s in outer_fb
    ok_inner = !used_by_inner || s in inner_fb
    if ok_outer && ok_inner
      push!(result, s)
    end
  end
  return Tuple(result)
end

function _partition(
  data,
  alg::NestedCV;
  target = nothing,
  time = nothing,
  groups = nothing,
  rng = Random.default_rng(),
  kwargs...,
)
  outer_cv = _partition(
    data,
    alg.outer;
    target = _slot_for(alg.outer, target, :target),
    time = _slot_for(alg.outer, time, :time),
    groups = _slot_for(alg.outer, groups, :groups),
    rng = rng,
  )

  outer_cv isa CrossValidationSplit || throw(
    SplitNotImplementedError(
      "NestedCV: outer strategy must return a CrossValidationSplit, got $(typeof(outer_cv)).",
    ),
  )
  all(f -> f isa TrainTestSplit, folds(outer_cv)) || throw(
    SplitNotImplementedError(
      "NestedCV: outer strategy must produce TrainTestSplit folds (no further nesting).",
    ),
  )

  nested = Vector{NestedFold{Vector{Int}}}(undef, length(folds(outer_cv)))
  for (i, fold) in enumerate(folds(outer_cv))
    train_pool = fold.train
    test_idx = fold.test

    data_inner = obsview(data, train_pool)
    inner_target = target === nothing ? nothing : view(target, train_pool)
    inner_time = time === nothing ? nothing : view(time, train_pool)
    inner_groups = groups === nothing ? nothing : view(groups, train_pool)

    inner_cv = partition(
      data_inner,
      alg.inner;
      target = inner_target,
      time = inner_time,
      groups = inner_groups,
      rng = rng,
    )

    inner_cv isa CrossValidationSplit || throw(
      SplitNotImplementedError(
        "NestedCV: inner strategy must return a CrossValidationSplit, got $(typeof(inner_cv)).",
      ),
    )

    remapped =
      [TrainTestSplit(train_pool[f.train], train_pool[f.test]) for f in folds(inner_cv)]
    nested[i] = NestedFold(train_pool, test_idx, CrossValidationSplit(remapped))
  end

  return CrossValidationSplit(nested)
end
