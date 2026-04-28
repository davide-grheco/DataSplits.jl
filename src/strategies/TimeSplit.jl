"""
    TimeSplit(order::Symbol=:asc) <: AbstractSplitStrategy

Splits a 1D array of dates/times into train/test sets, grouping by unique
values so that no group is split across train and test.

The actual training cohort size may slightly overshoot `n_train` but never
fall below it.

# Fields
- `order::Symbol`: `:asc` puts the oldest observations in train (default);
  `:desc` puts the newest in train.

# Examples
```julia
# dates is both data and the ordering variable
res = partition(dates, TimeSplitOldest(); train=70, test=30)
train_idx, test_idx = trainindices(res), testindices(res)

# X is the data to split; dates provides the ordering
res = partition(X, TimeSplitOldest(); time=dates, train=70, test=30)
X_train, X_test = splitdata(res, X)
```
"""
struct TimeSplit <: AbstractSplitStrategy
  order::Symbol
end

TimeSplit() = TimeSplit(:asc)

"""
    TimeSplitOldest()

Alias for `TimeSplit(:asc)` — oldest observations go to the training set.
"""
TimeSplitOldest() = TimeSplit(:asc)

"""
    TimeSplitNewest()

Alias for `TimeSplit(:desc)` — newest observations go to the training set.
"""
TimeSplitNewest() = TimeSplit(:desc)

consumes(::TimeSplit) = (:time,)
fallback_from_data(::TimeSplit) = (:time,)

function _partition(
  data,
  s::TimeSplit;
  time,
  n_train,
  n_test,
  rng = Random.GLOBAL_RNG,
  kwargs...,
)
  date_to_indices = Dict{eltype(time),Vector{Int}}()
  for (i, d) in enumerate(time)
    push!(get!(date_to_indices, d, Int[]), i)
  end
  dates = collect(keys(date_to_indices))
  descending = s.order in (:desc, :newest, :latest, :max, :maximum)
  sorted_dates = sort(dates; rev = descending)
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
