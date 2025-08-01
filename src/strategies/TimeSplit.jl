"""
    TimeSplit{T} <: SplitStrategy

Splits a 1D array of dates/times into train/test sets, grouping by unique date/time values. No group (samples with the same date) is split between train and test. The actual fraction may be slightly above the requested one, but never below.

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)
- `order::Symbol`: Sorting order; use `:asc` (oldest in train, default), `:desc` (newest in train)

# Returns
- `TrainTestSplit`: Indices for train and test sets.

# Examples
```julia
splitter = TimeSplitOldest(0.7)
result = split(dates, splitter)
train_idx, test_idx = splitdata(result, dates)
```
"""
struct TimeSplit{T} <: SplitStrategy
  frac::ValidFraction{T}
  order::Symbol
end

TimeSplit(frac::Real, order::Symbol = :asc) = TimeSplit(ValidFraction(frac), order)
TimeSplitOldest(frac::Real) = TimeSplit(frac, :asc)
TimeSplitNewest(frac::Real) = TimeSplit(frac, :desc)

function _split(data, s::TimeSplit; rng = Random.GLOBAL_RNG)
  N = numobs(data)
  date_to_indices = Dict{eltype(data),Vector{Int}}()
  for (i, d) in enumerate(data)
    push!(get!(date_to_indices, d, Int[]), i)
  end
  dates = collect(keys(date_to_indices))
  descending = s.order in (:desc, :newest, :latest, :max, :maximum)
  sorted_dates = sort(dates; rev = descending)

  total = 0
  n_train, n_test = train_test_counts(N, s.frac)
  train_idx = Int[]
  test_idx = Int[]
  for d in sorted_dates
    inds = date_to_indices[d]
    if total < n_train
      append!(train_idx, inds)
      total += length(inds)
    else
      append!(test_idx, inds)
    end
  end
  return TrainTestSplit(train_idx, test_idx)
end
