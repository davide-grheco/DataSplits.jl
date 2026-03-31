"""
    TimeSplit{T} <: AbstractSplitStrategy

Splits a 1D array of dates/times into train/test sets, grouping by unique
values so that no group is split across train and test.

The actual training fraction may be slightly above the requested one
but never below.

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)
- `order::Symbol`: `:asc` puts the oldest observations in train (default);
  `:desc` puts the newest in train.

# Examples
```julia
# dates is both data and the ordering variable
res = partition(dates, TimeSplitOldest(0.7))
train_idx, test_idx = trainindices(res), testindices(res)

# X is the data to split; dates provides the ordering
res = partition(X, TimeSplitOldest(0.7); time=dates)
X_train, X_test = splitdata(res, X)
```
"""
struct TimeSplit{T} <: AbstractSplitStrategy
  frac::ValidFraction{T}
  order::Symbol
end

TimeSplit(frac::Real, order::Symbol = :asc) = TimeSplit(ValidFraction(frac), order)

"""
    TimeSplitOldest(frac)

Alias for `TimeSplit(frac, :asc)` — oldest observations go to the training set.
"""
TimeSplitOldest(frac::Real) = TimeSplit(frac, :asc)

"""
    TimeSplitNewest(frac)

Alias for `TimeSplit(frac, :desc)` — newest observations go to the training set.
"""
TimeSplitNewest(frac::Real) = TimeSplit(frac, :desc)

consumes(::TimeSplit) = (:time,)
fallback_from_data(::TimeSplit) = (:time,)

function _partition(data, s::TimeSplit; time, rng = Random.GLOBAL_RNG, kwargs...)
  N = numobs(data)
  date_to_indices = Dict{eltype(time),Vector{Int}}()
  for (i, d) in enumerate(time)
    push!(get!(date_to_indices, d, Int[]), i)
  end
  dates = collect(keys(date_to_indices))
  descending = s.order in (:desc, :newest, :latest, :max, :maximum)
  sorted_dates = sort(dates; rev = descending)
  n_train, _ = train_test_counts(N, s.frac)
  total = 0
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
