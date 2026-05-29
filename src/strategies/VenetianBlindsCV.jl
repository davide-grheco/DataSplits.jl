"""
    VenetianBlindsCV(k; shuffle=false) <: AbstractCVStrategy

Venetian-blinds k-fold cross-validation.

Sorts samples by `target` (ascending), then assigns them to folds in a
round-robin pattern: sample ranked 1st → fold 1, 2nd → fold 2, …, k-th → fold k,
(k+1)-th → fold 1, and so on.

Unlike [`KFold`](@ref), which assigns *contiguous blocks* to each fold,
Venetian Blinds distributes the sorted sequence evenly, so every fold covers
the full value range.  Unlike [`StratifiedKFold`](@ref), it sorts globally
with no binning step — the assignment is deterministic and does not require
choosing a number of quantile bins.

# Fields
- `k::Int`: Number of folds (must be ≥ 2 and ≤ N).
- `shuffle::Bool`: When `true`, samples with identical target values (ties)
  are randomly permuted within their tied group before fold assignment, using
  the `rng` passed to `partition`.  When `false` (default), ties are broken
  by original sample index (stable sort).

# Notes
- `target` falls back to `data` when omitted: `partition(y, VenetianBlindsCV(5))`
  sorts directly by the data vector.
- For natural index-order (no sorting), pass `target = collect(1:numobs(data))`.

# Examples
```julia
# Sort by target value — each fold spans the full response range.
cvs = partition(X, VenetianBlindsCV(5); target = y)

# Directly on a target vector (no X needed).
cvs = partition(y, VenetianBlindsCV(5))

# Randomise tie-breaking.
cvs = partition(X, VenetianBlindsCV(5; shuffle = true); target = y,
                rng = MersenneTwister(42))
```

# See also
[`KFold`](@ref), [`StratifiedKFold`](@ref)

# References
Naes, T.; Isaksson, T.; Fearn, T.; Davies, T. *A User-Friendly Guide to Multivariate
Calibration and Classification*. NIR Publications, 2002.
"""
struct VenetianBlindsCV <: AbstractCVStrategy
  k::Int
  shuffle::Bool
end

VenetianBlindsCV(k::Integer; shuffle::Bool = false) = VenetianBlindsCV(Int(k), shuffle)

consumes(::VenetianBlindsCV) = (:target,)
fallback_from_data(::VenetianBlindsCV) = (:target,)

# Shuffle within runs of equal values in target[order] (stable-sort tie groups).
function _venetian_shuffle_ties!(rng, order::AbstractVector{Int}, target::AbstractVector)
  n = length(order)
  i = 1
  while i <= n
    j = i
    while j < n && target[order[j+1]] == target[order[i]]
      j += 1
    end
    j > i && shuffle!(rng, view(order, i:j))
    i = j + 1
  end
end

function _partition(
  data,
  alg::VenetianBlindsCV;
  target,
  rng = Random.default_rng(),
  kwargs...,
)
  alg.k >= 2 ||
    throw(SplitParameterError("VenetianBlindsCV requires k ≥ 2, got k=$(alg.k)."))
  N = numobs(data)
  alg.k <= N ||
    throw(SplitParameterError("VenetianBlindsCV requires k ≤ N=$N, got k=$(alg.k)."))

  order = sortperm(target)
  alg.shuffle && _venetian_shuffle_ties!(rng, order, target)

  fold_test = [Int[] for _ = 1:alg.k]
  for (i, idx) in enumerate(order)
    push!(fold_test[mod1(i, alg.k)], idx)
  end

  result = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.k)
  for f = 1:alg.k
    result[f] = TrainTestSplit(setdiff(1:N, fold_test[f]), fold_test[f])
  end
  return CrossValidationSplit(result)
end
