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
  rng = Random.default_rng(),
  kwargs...,
)
  sorted_keys, order = groupsortperm(time)
  block_offset = group_offsets(sorted_keys, order, time)
  descending = s.order in (:desc, :newest, :latest, :max, :maximum)
  block_range = descending ? (length(sorted_keys):-1:1) : (1:length(sorted_keys))
  total = 0
  train_idx = Int[]
  test_idx = Int[]
  for b in block_range
    inds = order[(block_offset[b]+1):block_offset[b+1]]
    if total < n_train
      append!(train_idx, inds)
      total += length(inds)
    else
      append!(test_idx, inds)
    end
  end
  return TrainTestSplit(train_idx, test_idx)
end
