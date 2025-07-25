"""
    TargetPropertySplit{T} <: SplitStrategy

A splitting strategy that partitions a 1D property array into train/test sets by sorting the property values.

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)
- `order::Symbol`: Sorting order; use `:asc`, `:desc`, `:high`, `:low`, `:largest`, `:smallest`, etc.

# Examples
```julia
split(y, TargetPropertyHigh(0.8))
split(X[:, 3], TargetPropertyLow(0.5))
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
function targetpropertysplit(N, s, rng, data::AbstractVector)
  idx = collect(1:N)
  pairs = collect(zip(data, idx))
  descending = s.order in (:desc, :high, :largest, :max, :maximum)
  sorted_pairs = sort(pairs; by = first, rev = descending)
  n_train = floor(Int, s.frac * N)
  train_pos = [p[2] for p in sorted_pairs[1:n_train]]
  test_pos = [p[2] for p in sorted_pairs[n_train+1:end]]
  return train_pos, test_pos
end

function _split(data, s::TargetPropertySplit; rng = Random.GLOBAL_RNG)
  split_with_positions(data, s, targetpropertysplit; rng = rng)
end
