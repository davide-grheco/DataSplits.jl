using Random

"""
    GroupShuffleSplitCV(n_splits::Integer) <: AbstractResamplingCVStrategy

Group-aware random permutation cross-validation. For each of `n_splits`
iterations the groups are shuffled and assigned whole into the train cohort
until the requested training size is reached; the remaining groups form the
test cohort. Mirrors scikit-learn's `GroupShuffleSplit`.

Groups are passed as a vector of membership IDs via the `groups=` keyword
(or, by fallback, `data` itself plays that role).

# Fields
- `n_splits::Int`: Number of resamples (must be ≥ 1).

# Notes
- **Resamples are independent** — a group can land in train in one fold and
  test in another. This is the defining property versus [`GroupKFold`](@ref),
  where each group appears in exactly one test cohort across folds.
- Because groups are added whole, the actual train cohort size may overshoot
  the requested `n_train` (same behaviour as the 2-cohort
  [`GroupShuffleSplit`](@ref)).
- `train` and `test` must sum to `N`; every observation is placed in
  exactly one cohort per resample.

# Examples
```julia
# Fractions.
cvs = partition(X, GroupShuffleSplitCV(10);
                groups = patient_ids, train = 0.8, test = 0.2)

# Absolute counts.
cvs = partition(X, GroupShuffleSplitCV(10);
                groups = patient_ids, train = 80, test = 20)

# Reproducible.
cvs = partition(X, GroupShuffleSplitCV(10);
                groups = patient_ids, train = 0.8, test = 0.2,
                rng = MersenneTwister(42))

# Fallback: ids are simultaneously the data and the groups.
cvs = partition(patient_ids, GroupShuffleSplitCV(10);
                train = 0.8, test = 0.2)
```
"""
struct GroupShuffleSplitCV <: AbstractResamplingCVStrategy
  n_splits::Int
end

GroupShuffleSplitCV(n_splits::Integer) = GroupShuffleSplitCV(Int(n_splits))

consumes(::GroupShuffleSplitCV) = (:groups,)
fallback_from_data(::GroupShuffleSplitCV) = (:groups,)

function _partition(
  data,
  alg::GroupShuffleSplitCV;
  groups,
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  alg.n_splits >= 1 || throw(
    SplitParameterError(
      "GroupShuffleSplitCV requires n_splits ≥ 1, got n_splits=$(alg.n_splits).",
    ),
  )

  sorted_keys, perm = groupsortperm(groups)
  off = group_offsets(sorted_keys, perm, groups)
  n_groups = length(sorted_keys)

  result = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.n_splits)
  for i = 1:alg.n_splits
    block_order = shuffle(rng, collect(1:n_groups))
    train_pos = Int[]
    test_pos = Int[]
    for b in block_order
      idxs = perm[(off[b]+1):off[b+1]]
      if length(train_pos) < n_train
        append!(train_pos, idxs)
      else
        append!(test_pos, idxs)
      end
    end
    result[i] = TrainTestSplit(train_pos, test_pos)
  end
  return CrossValidationSplit(result)
end
