"""
    TargetPropertySplit(order::Symbol) <: AbstractSplitStrategy

Splits observations by sorting a 1D property vector and selecting the top or
bottom slice as the training set.

# Fields
- `order::Symbol`: `:high` selects the largest values for training; `:low` selects the smallest.

# Examples
```julia
# y is both data and the property to sort by
res = partition(y, TargetPropertyHigh(); train=80, test=20)
y_train, y_test = splitdata(res, y)

# X is the data to split; y is the property
res = partition(X, TargetPropertyHigh(); target=y, train=80, test=20)
X_train, X_test = splitdata(res, X)
```
"""
struct TargetPropertySplit <: AbstractSplitStrategy
  order::Symbol
end

"""
    TargetPropertyHigh()

Alias for `TargetPropertySplit(:high)` — selects the highest-valued
observations for the training set.
"""
TargetPropertyHigh() = TargetPropertySplit(:high)

"""
    TargetPropertyLow()

Alias for `TargetPropertySplit(:low)` — selects the lowest-valued
observations for the training set.
"""
TargetPropertyLow() = TargetPropertySplit(:low)

consumes(::TargetPropertySplit) = (:target,)
fallback_from_data(::TargetPropertySplit) = (:target,)

function _partition(
  data,
  s::TargetPropertySplit;
  target,
  n_train,
  n_test,
  rng = Random.GLOBAL_RNG,
  kwargs...,
)
  descending = s.order in (:desc, :high, :largest, :max, :maximum)
  idx = sortperm(target; rev = descending)
  return TrainTestSplit(idx[1:n_train], idx[n_train+1:end])
end
