using Combinatorics: combinations

"""
    CombinatorialPurgedKFold(k, n_test_folds; purge=0, embargo=0) <: AbstractCVStrategy

Combinatorial Purged k-fold cross-validation (CPCV).

Generalises [`PurgedKFold`](@ref) by exhaustively testing all possible
combinations of `n_test_folds` out of `k` time-ordered folds, rather than
testing each fold exactly once.  This produces `C(k, n_test_folds)` train/test
pairs, each independently purged and embarked, eliminating the path-dependency
of standard walk-forward validation.

Each test set is the union of `n_test_folds` contiguous time blocks; each train
set is the complement of the test set and its exclusion windows.  Purging and
embargo work identically to `PurgedKFold`: `purge` observations immediately
**before** each test block are removed from train, and `embargo` observations
immediately **after** each test block are removed.

# Fields
- `k::Int`: Number of time-ordered folds (must be ≥ 2).
- `n_test_folds::Int`: Number of folds in each test set (must be ≥ 1 and < k).
- `purge::Int`: Observations removed from train immediately before each test
  block (default `0`).
- `embargo::Int`: Observations removed from train immediately after each test
  block (default `0`).

# Notes
- `n_test_folds = 1` is equivalent to [`PurgedKFold`](@ref) (same k folds,
  same purge/embargo).
- The number of resulting folds grows combinatorially: `C(k, n_test_folds)`.
  For `k=6, n_test_folds=2` this gives 15 folds; for `k=8, n_test_folds=3`,
  56 folds.
- Average performance across all folds to obtain a path-independent estimate.

# Examples
```julia
# 6 time blocks, always test on 2 simultaneously → 15 train/test pairs.
cvs = partition(X, CombinatorialPurgedKFold(6, 2; purge=1, embargo=1);
                time = timestamps)

# Equivalent to PurgedKFold(5; purge=1) when n_test_folds=1.
cvs = partition(X, CombinatorialPurgedKFold(5, 1; purge=1); time = timestamps)
```

# References
López de Prado, M. *Advances in Financial Machine Learning*. Wiley, 2018,
§12 ("Backtesting through Cross-Validation").
"""
struct CombinatorialPurgedKFold <: AbstractCVStrategy
  k::Int
  n_test_folds::Int
  purge::Int
  embargo::Int
end

function CombinatorialPurgedKFold(
  k::Integer,
  n_test_folds::Integer;
  purge::Integer = 0,
  embargo::Integer = 0,
)
  k >= 2 || throw(SplitParameterError("CombinatorialPurgedKFold requires k ≥ 2, got k=$k."))
  1 <= n_test_folds < k || throw(
    SplitParameterError(
      "CombinatorialPurgedKFold requires 1 ≤ n_test_folds < k, got n_test_folds=$n_test_folds.",
    ),
  )
  purge >= 0 || throw(
    SplitParameterError("CombinatorialPurgedKFold requires purge ≥ 0, got purge=$purge."),
  )
  embargo >= 0 || throw(
    SplitParameterError(
      "CombinatorialPurgedKFold requires embargo ≥ 0, got embargo=$embargo.",
    ),
  )
  CombinatorialPurgedKFold(Int(k), Int(n_test_folds), Int(purge), Int(embargo))
end

consumes(::CombinatorialPurgedKFold) = (:time,)
fallback_from_data(::CombinatorialPurgedKFold) = (:time,)

function _partition(data, alg::CombinatorialPurgedKFold; time, kwargs...)
  N = numobs(data)
  sorted_dates, order = groupsortperm(time)
  B = length(sorted_dates)

  alg.k <= B || throw(
    SplitParameterError(
      "CombinatorialPurgedKFold(k=$(alg.k)) requires at least k=$(alg.k) distinct time values; got $B.",
    ),
  )

  chunk_block_end = distribute_blocks(B, alg.k)
  block_offset = group_offsets(sorted_dates, order, time)

  # Position bounds (in sorted order) for each of the k blocks
  block_lo = Vector{Int}(undef, alg.k)
  block_hi = Vector{Int}(undef, alg.k)
  for i = 1:alg.k
    start_block = i == 1 ? 1 : chunk_block_end[i-1] + 1
    block_lo[i] = block_offset[start_block] + 1
    block_hi[i] = block_offset[chunk_block_end[i]+1]
  end

  result = TrainTestSplit{Vector{Int}}[]

  for test_blocks in combinations(1:alg.k, alg.n_test_folds)
    # Test indices: union of all positions in the selected blocks
    test_positions = Int[]
    for b in test_blocks
      append!(test_positions, block_lo[b]:block_hi[b])
    end
    test_idx = order[test_positions]

    # Exclusion set: test + purge windows before each test block
    #                          + embargo windows after each test block
    excluded = Set{Int}(test_idx)
    for b in test_blocks
      for pos = max(1, block_lo[b]-alg.purge):(block_lo[b]-1)
        push!(excluded, order[pos])
      end
      for pos = (block_hi[b]+1):min(N, block_hi[b]+alg.embargo)
        push!(excluded, order[pos])
      end
    end

    train_idx = [i for i = 1:N if i ∉ excluded]
    isempty(train_idx) && throw(
      SplitParameterError(
        "CombinatorialPurgedKFold: a fold has an empty train cohort " *
        "(purge=$(alg.purge), embargo=$(alg.embargo) too large for the surrounding blocks).",
      ),
    )

    push!(result, TrainTestSplit(train_idx, test_idx))
  end

  return CrossValidationSplit(result)
end
