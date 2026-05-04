using Combinatorics: combinations

"""
    LeavePGroupsOut(p::Integer) <: AbstractCVStrategy

Leave-p-groups-out cross-validation. Produces one fold per combination of
`p` distinct groups: in each fold, the test cohort is exactly the
observations belonging to those `p` groups, and the train cohort is
everything else.

Equivalent to scikit-learn's `LeavePGroupsOut`. The number of folds is
`binomial(n_groups, p)`, which grows quickly — pick `p` accordingly.

Groups are passed via the `groups=` keyword (or, by fallback, `data` itself
plays that role).

# Fields
- `p::Int`: Number of groups held out as test in each fold (must satisfy
  `1 ≤ p < n_groups`).

# Constructors
- `LeavePGroupsOut(p)` — generic constructor.
- `LeaveOneGroupOut()` — convenience alias for `LeavePGroupsOut(1)`.

# Examples
```julia
# One group out per fold (n_folds == n_groups).
cvs = partition(X, LeaveOneGroupOut(); groups = patient_ids)

# Two groups out per fold; n_folds == binomial(n_groups, 2).
cvs = partition(X, LeavePGroupsOut(2); groups = site_ids)

for (X_train, X_test) in splitview(cvs, X)
    # train and evaluate
end
```
"""
struct LeavePGroupsOut <: AbstractCVStrategy
  p::Int
end

LeavePGroupsOut(p::Integer) = LeavePGroupsOut(Int(p))

"""
    LeaveOneGroupOut()

Alias for `LeavePGroupsOut(1)` — produces one fold per unique group.
"""
LeaveOneGroupOut() = LeavePGroupsOut(1)

consumes(::LeavePGroupsOut) = (:groups,)
fallback_from_data(::LeavePGroupsOut) = (:groups,)

function _partition(data, alg::LeavePGroupsOut; groups, kwargs...)
  alg.p >= 1 ||
    throw(SplitParameterError("LeavePGroupsOut requires p ≥ 1, got p=$(alg.p)."))

  N = numobs(data)
  # Anticipates issue #22 (length validation belongs in `partition`).
  length(groups) == N || throw(
    SplitInputError(
      "`groups` length ($(length(groups))) does not match number of observations ($N).",
    ),
  )

  group_to_indices = Dict{eltype(groups),Vector{Int}}()
  group_order = eltype(groups)[]
  for (i, g) in enumerate(groups)
    if !haskey(group_to_indices, g)
      push!(group_order, g)
      group_to_indices[g] = Int[]
    end
    push!(group_to_indices[g], i)
  end

  n_groups = length(group_order)
  alg.p < n_groups || throw(
    SplitParameterError(
      "LeavePGroupsOut(p=$(alg.p)) requires p < n_groups; got n_groups=$n_groups (would leave the train cohort empty).",
    ),
  )

  folds = map(combinations(group_order, alg.p)) do test_groups_combo
    test_groups = Set(test_groups_combo)
    test_idx = Int[]
    train_idx = Int[]
    for i = 1:N
      if groups[i] in test_groups
        push!(test_idx, i)
      else
        push!(train_idx, i)
      end
    end
    TrainTestSplit(train_idx, test_idx)
  end
  return CrossValidationSplit(folds)
end
