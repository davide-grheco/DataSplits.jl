using MLUtils: obsview

"""
    WithValidation(test_strategy, val_strategy) <: AbstractSplitStrategy

Combinator producing a train/validation/test split by composing two existing
strategies:

1. `test_strategy` is applied to the full dataset, separating the test set
   from the train pool.
2. `val_strategy` is applied to the train pool, separating the validation
   set from the final training set.

Both inner strategies must return a [`TrainTestSplit`](@ref). Each `frac` is
interpreted relative to its own input, consistent with the rest of the
package. The total fractions therefore combine multiplicatively — e.g.
`WithValidation(KennardStoneSplit(0.8), KennardStoneSplit(0.8))` yields
64 % train, 16 % validation and 20 % test.

The slot interface ([`consumes`](@ref) / [`fallback_from_data`](@ref)) is
the union of both inner strategies' slots, so any combination of `:data`,
`:target`, `:time` and `:groups` consumers is supported.

# Fields
- `test_strategy::AbstractSplitStrategy`: produces the train_pool/test split.
- `val_strategy::AbstractSplitStrategy`: produces the train/val split inside
  the train pool.

# Examples
```julia
# Same strategy on both passes
res = partition(X, WithValidation(KennardStoneSplit(0.8), KennardStoneSplit(0.8)))
X_train, X_val, X_test = splitdata(res, X)

# Mix slots: time-aware test split, target-aware validation split
res = partition(
  X,
  WithValidation(TimeSplitOldest(0.8), TargetPropertyHigh(0.8));
  time = dates,
  target = y,
)
```

# See also
[`TrainValTestSplit`](@ref), [`partition`](@ref).
"""
struct WithValidation{T<:AbstractSplitStrategy,V<:AbstractSplitStrategy} <:
       AbstractSplitStrategy
  test_strategy::T
  val_strategy::V
end

_union_slots(a::Tuple, b::Tuple) = (unique((a..., b...))...,)

consumes(w::WithValidation) = _union_slots(consumes(w.test_strategy), consumes(w.val_strategy))

fallback_from_data(w::WithValidation) =
  _union_slots(fallback_from_data(w.test_strategy), fallback_from_data(w.val_strategy))

function _partition(
  data,
  w::WithValidation;
  target = nothing,
  time = nothing,
  groups = nothing,
  rng = Random.default_rng(),
  kwargs...,
)
  outer = _partition(
    data,
    w.test_strategy;
    target = target,
    time = time,
    groups = groups,
    rng = rng,
  )
  outer isa TrainTestSplit || throw(
    SplitNotImplementedError(
      "WithValidation requires the test strategy to return a TrainTestSplit, got $(typeof(outer)).",
    ),
  )

  train_pool = outer.train
  test_idx = outer.test

  data_pool = obsview(data, train_pool)
  target_pool = target === nothing ? nothing : view(target, train_pool)
  time_pool = time === nothing ? nothing : view(time, train_pool)
  groups_pool = groups === nothing ? nothing : view(groups, train_pool)

  inner = _partition(
    data_pool,
    w.val_strategy;
    target = target_pool,
    time = time_pool,
    groups = groups_pool,
    rng = rng,
  )
  inner isa TrainTestSplit || throw(
    SplitNotImplementedError(
      "WithValidation requires the validation strategy to return a TrainTestSplit, got $(typeof(inner)).",
    ),
  )

  train_idx = train_pool[inner.train]
  val_idx = train_pool[inner.test]
  return TrainValTestSplit(train_idx, val_idx, test_idx)
end
