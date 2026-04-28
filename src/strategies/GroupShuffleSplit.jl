using Random

"""
    GroupShuffleSplit() <: AbstractSplitStrategy

Group-aware train/test splitting. Accumulates whole groups into the training
set (in random order) until the requested training size is reached.

Groups are passed as a vector of membership IDs via the `groups=` keyword.
Any grouping is valid: cluster assignments, patient IDs, scaffold labels,
batch numbers, site identifiers, graph communities, etc.

# Notes
Because groups are added whole, the actual train cohort size may overshoot
the requested `n_train`. No attempt is made to minimise this overshoot.

# Examples
```julia
# ids is both data and groups
res = partition(ids, GroupShuffleSplit(); train=80, test=20)

# X is split; group membership provided separately
res = partition(X, GroupShuffleSplit(); groups=patient_ids, train=80, test=20)
X_train, X_test = splitdata(res, X)

# With Clustering.jl
using Clustering
res = partition(X, GroupShuffleSplit();
                groups=assignments(kmeans(X, 5)), train=80, test=20)
```
"""
struct GroupShuffleSplit <: AbstractSplitStrategy end

consumes(::GroupShuffleSplit) = (:groups,)
fallback_from_data(::GroupShuffleSplit) = (:groups,)

function _partition(
  data,
  ::GroupShuffleSplit;
  groups,
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  cl_ids = unique(groups)
  shuffle!(rng, cl_ids)
  train_pos = Int[]
  test_pos = Int[]
  for cid in cl_ids
    idxs = findall(==(cid), groups)
    if length(train_pos) < n_train
      append!(train_pos, idxs)
    else
      append!(test_pos, idxs)
    end
  end
  return TrainTestSplit(train_pos, test_pos)
end
