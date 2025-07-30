"""
    TargetPropertySplit{T} <: SplitStrategy

Splits a 1D property array into train/test sets by sorting the property values.

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)
- `order::Symbol`: Sorting order; use `:asc`, `:desc`, `:high`, `:low`, `:largest`, `:smallest`, etc.

# Returns
- `TrainTestSplit`: Indices for train and test sets.

# Examples
```julia
splitter = TargetPropertyHigh(0.8)
result = split(y, splitter)
train_idx, test_idx = splitdata(result, y)
```
"""
struct TargetPropertySplit{T} <: SplitStrategy
  frac::ValidFraction{T}
  order::Symbol
end

TargetPropertySplit(frac::Real, order::Symbol) =
  TargetPropertySplit(ValidFraction(frac), order)
TargetPropertyHigh(frac::Real) = TargetPropertySplit(frac, :high)
TargetPropertyLow(frac::Real) = TargetPropertySplit(frac, :low)

"""
    targetpropertysplit(N, s, rng, data)

Sorts the property array and splits into train/test according to `s.frac` and `s.order`.
"""
function _split(data, s::TargetPropertySplit; rng = Random.GLOBAL_RNG)
  N = numobs(data)
  idx = collect(1:N)
  pairs = collect(zip(data, idx))
  descending = s.order in (:desc, :high, :largest, :max, :maximum)
  sorted_pairs = sort(pairs; by = first, rev = descending)
  n_train, n_test = train_test_counts(N, s.frac)
  train_pos = [p[2] for p in sorted_pairs[1:n_train]]
  test_pos = [p[2] for p in sorted_pairs[n_train+1:end]]
  return TrainTestSplit(train_pos, test_pos)
end
