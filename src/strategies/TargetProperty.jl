"""
    TargetPropertySplit{T} <: AbstractSplitStrategy

Splits observations by sorting a 1D property vector and selecting the top or
bottom fraction as the training set.

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)
- `order::Symbol`: `:high` selects the largest values for training; `:low` selects the smallest.

# Examples
```julia
# y is both data and the property to sort by
res = partition(y, TargetPropertyHigh(0.8))
y_train, y_test = splitdata(res, y)

# X is the data to split; y is the property
res = partition(X, TargetPropertyHigh(0.8); target=y)
X_train, X_test = splitdata(res, X)
```
"""
struct TargetPropertySplit{T} <: AbstractSplitStrategy
  frac::ValidFraction{T}
  order::Symbol
end

TargetPropertySplit(frac::Real, order::Symbol) =
  TargetPropertySplit(ValidFraction(frac), order)

"""
    TargetPropertyHigh(frac)

Alias for `TargetPropertySplit(frac, :high)` — selects the highest-valued
observations for the training set.
"""
TargetPropertyHigh(frac::Real) = TargetPropertySplit(frac, :high)

"""
    TargetPropertyLow(frac)

Alias for `TargetPropertySplit(frac, :low)` — selects the lowest-valued
observations for the training set.
"""
TargetPropertyLow(frac::Real) = TargetPropertySplit(frac, :low)

consumes(::TargetPropertySplit) = (:target,)
fallback_from_data(::TargetPropertySplit) = (:target,)

function _partition(
  data,
  s::TargetPropertySplit;
  target,
  rng = Random.GLOBAL_RNG,
  kwargs...,
)
  N = numobs(data)
  descending = s.order in (:desc, :high, :largest, :max, :maximum)
  idx = sortperm(target; rev = descending)
  n_train, _ = train_test_counts(N, s.frac)
  return TrainTestSplit(idx[1:n_train], idx[n_train+1:end])
end
