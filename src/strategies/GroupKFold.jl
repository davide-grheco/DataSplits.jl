"""
    GroupKFold(k::Integer; shuffle::Bool=false) <: AbstractCVStrategy

Group-aware k-fold cross-validation. Whole groups are assigned to a single
fold; no group ever appears in both the train and test cohort of the same
fold. Equivalent in spirit to scikit-learn's `GroupKFold`.

Groups are passed via the `groups=` keyword (or, by fallback, `data` itself
plays that role).

# Fields
- `k::Int`: Number of folds (must be ≥ 2 and ≤ number of unique groups).
- `shuffle::Bool`: When `true`, the order in which groups are considered
  for fold assignment is shuffled using the `rng` passed to `partition`,
  so different RNG seeds yield different fold compositions. When `false`
  (default), assignment is deterministic and reproducible without an `rng`.

# Notes
- Within each candidate group, the algorithm places it in the currently
  smallest fold, so observation counts across folds stay roughly balanced
  whether or not shuffling is enabled. Mirrors sklearn's `GroupKFold`.
- When `shuffle=false`, groups are processed in **descending order of size**
  (the classic deterministic balancing). When `shuffle=true`, they are
  processed in a randomly permuted order — folds remain balanced but no
  longer follow size order.

# Examples
```julia
# Deterministic (default).
cvs = partition(X, GroupKFold(5); groups = patient_ids)

# Shuffled — different seeds give different fold compositions.
cvs = partition(X, GroupKFold(5; shuffle = true);
                groups = patient_ids, rng = MersenneTwister(42))

for (X_train, X_test) in splitview(cvs, X)
    # train and evaluate
end

# Fallback: ids are simultaneously the data and the groups.
cvs = partition(patient_ids, GroupKFold(5))
```
"""
struct GroupKFold <: AbstractCVStrategy
  k::Int
  shuffle::Bool
end

GroupKFold(k::Integer; shuffle::Bool = false) = GroupKFold(Int(k), shuffle)

consumes(::GroupKFold) = (:groups,)
fallback_from_data(::GroupKFold) = (:groups,)

function _partition(data, alg::GroupKFold; groups, rng = Random.default_rng(), kwargs...)
  alg.k >= 2 || throw(SplitParameterError("GroupKFold requires k ≥ 2, got k=$(alg.k)."))

  N = numobs(data)
  sorted_keys, perm = groupsortperm(groups)
  off = group_offsets(sorted_keys, perm, groups)
  n_groups = length(sorted_keys)
  alg.k <= n_groups || throw(
    SplitParameterError(
      "GroupKFold(k=$(alg.k)) requires at least k unique groups; got $n_groups.",
    ),
  )

  block_order = collect(1:n_groups)
  if alg.shuffle
    Random.shuffle!(rng, block_order)
  else
    sort!(block_order; by = b -> -(off[b+1] - off[b]))
  end

  fold_of_block = Vector{Int}(undef, n_groups)
  fold_counts = zeros(Int, alg.k)
  for b in block_order
    f = argmin(fold_counts)
    fold_of_block[b] = f
    fold_counts[f] += off[b+1] - off[b]
  end

  fold_test = [Int[] for _ = 1:alg.k]
  for b = 1:n_groups
    append!(fold_test[fold_of_block[b]], perm[(off[b]+1):off[b+1]])
  end

  folds = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.k)
  for f = 1:alg.k
    folds[f] = TrainTestSplit(setdiff(1:N, fold_test[f]), fold_test[f])
  end
  return CrossValidationSplit(folds)
end
