using Random

"""
    GroupShuffleSplit(frac::Real) <: AbstractSplitStrategy

Group-aware train/test splitting. Accumulates whole groups into the training
set (in random order) until the requested fraction is reached.

Groups are passed as a vector of membership IDs via the `groups=` keyword.
Any grouping is valid: cluster assignments, patient IDs, scaffold labels,
batch numbers, site identifiers, graph communities, etc.

# Fields
- `frac::ValidFraction`: Fraction of samples in the training set (0 < frac < 1)

# Notes
Because groups are added whole, the actual training fraction may overshoot
the requested value. No attempt is made to minimise this overshoot.

# Examples
```julia
# ids is both data and groups
res = partition(ids, GroupShuffleSplit(0.8))

# X is split; group membership provided separately
res = partition(X, GroupShuffleSplit(0.8); groups=patient_ids)
X_train, X_test = splitdata(res, X)

# With Clustering.jl
using Clustering
res = partition(X, GroupShuffleSplit(0.8); groups=assignments(kmeans(X, 5)))
```
"""
struct GroupShuffleSplit <: AbstractSplitStrategy
  frac::ValidFraction
end

GroupShuffleSplit(frac::Real) = GroupShuffleSplit(ValidFraction(frac))

consumes(::GroupShuffleSplit) = (:groups,)
fallback_from_data(::GroupShuffleSplit) = (:groups,)

function _partition(
  data,
  s::GroupShuffleSplit;
  groups,
  rng = Random.default_rng(),
  kwargs...,
)
  N = numobs(data)
  cl_ids = unique(groups)
  shuffle!(rng, cl_ids)
  train_pos = Int[]
  test_pos = Int[]
  for cid in cl_ids
    idxs = findall(==(cid), groups)
    if length(train_pos) / N < float(s.frac)
      append!(train_pos, idxs)
    else
      append!(test_pos, idxs)
    end
  end
  return TrainTestSplit(train_pos, test_pos)
end
