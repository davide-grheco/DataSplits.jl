"""
    BlockedCV(k::Integer; gap::Integer=0) <: AbstractCVStrategy

Blocked k-fold cross-validation for dependent (time- or space-ordered)
data. Observations are sorted by `time=` and partitioned into `k`
contiguous chronological blocks; each block takes a turn as the test
cohort while the train cohort is everything **else** — both blocks
preceding it and blocks following it.

This differs from [`TimeSeriesSplit`](@ref) (forward-only, train always
precedes test) and from [`KFold`](@ref) (no temporal ordering). It
matches the "blocked CV" used in time-series / spatial-statistics
literature (Bergmeir & Benítez 2012, Roberts et al. 2017) when train
samples should not be drawn from arbitrary positions but the test
block must still be embedded in a longer train history.

A `gap` window (in observations) is removed from the train cohort on
**both sides** of the test block to mitigate leakage from
autocorrelation.

# Atomicity rule

Observations sharing the same timestamp are never split between train
and test of the same fold — block boundaries fall between distinct
time values, mirroring [`TimeSeriesSplit`](@ref) and [`TimeSplit`](@ref).
`gap` is measured in observations and may trim partial blocks of equal
timestamps from the train side; no row ever leaks into test.

# Fields
- `k::Int`: Number of folds (must be ≥ 2 and ≤ number of distinct
  time values).
- `gap::Int`: Number of observations excluded from the train cohort
  on each side of the test block (must be ≥ 0; default `0`).

# Examples
```julia
# 5-fold blocked CV with a one-observation embargo on both sides.
cvs = partition(X, BlockedCV(5; gap = 1); time = timestamps)

for (X_train, X_test) in splitview(cvs, X)
    fit!(model, X_train); evaluate(model, X_test)
end

# Single-input shorthand when the timestamps are also the data.
cvs = partition(timestamps, BlockedCV(4))
```
"""
struct BlockedCV <: AbstractCVStrategy
  k::Int
  gap::Int
end

BlockedCV(k::Integer; gap::Integer = 0) = BlockedCV(Int(k), Int(gap))

consumes(::BlockedCV) = (:time,)
fallback_from_data(::BlockedCV) = (:time,)

function _partition(data, alg::BlockedCV; time, kwargs...)
  alg.k >= 2 ||
    throw(SplitParameterError("BlockedCV requires k ≥ 2, got k=$(alg.k)."))
  alg.gap >= 0 ||
    throw(SplitParameterError("BlockedCV requires gap ≥ 0, got gap=$(alg.gap)."))

  N = numobs(data)

  # Group observations by identical timestamp (atomic time blocks).
  date_to_indices = Dict{eltype(time),Vector{Int}}()
  for (i, t) in enumerate(time)
    push!(get!(date_to_indices, t, Int[]), i)
  end
  sorted_dates = sort!(collect(keys(date_to_indices)))
  B = length(sorted_dates)

  alg.k <= B || throw(
    SplitParameterError(
      "BlockedCV(k=$(alg.k)) requires at least k distinct time values; got $B.",
    ),
  )

  # Distribute the B blocks across k chunks balancing the remainder across
  # the first `B mod k` chunks (np.array_split behaviour).
  base, rem = divrem(B, alg.k)
  chunk_block_end = Vector{Int}(undef, alg.k)
  acc = 0
  for c = 1:alg.k
    acc += base + (c <= rem ? 1 : 0)
    chunk_block_end[c] = acc
  end

  # Chronological order vector and block→position offsets.
  order = Int[]
  block_offset = Vector{Int}(undef, B + 1)
  block_offset[1] = 0
  for (b, d) in enumerate(sorted_dates)
    inds = date_to_indices[d]
    append!(order, inds)
    block_offset[b+1] = block_offset[b] + length(inds)
  end

  result = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.k)
  for i = 1:alg.k
    test_block_start = i == 1 ? 1 : chunk_block_end[i-1] + 1
    test_block_end = chunk_block_end[i]

    test_lo = block_offset[test_block_start] + 1
    test_hi = block_offset[test_block_end+1]

    # Train side: everything outside [test_lo - gap, test_hi + gap].
    train_left_hi = test_lo - 1 - alg.gap
    train_right_lo = test_hi + 1 + alg.gap

    train_left = train_left_hi >= 1 ? order[1:train_left_hi] : Int[]
    train_right = train_right_lo <= N ? order[train_right_lo:N] : Int[]
    train_idx = vcat(train_left, train_right)

    !isempty(train_idx) || throw(
      SplitParameterError(
        "BlockedCV: fold $i has empty train cohort (gap=$(alg.gap) too large for the surrounding blocks).",
      ),
    )

    result[i] = TrainTestSplit(train_idx, order[test_lo:test_hi])
  end
  return CrossValidationSplit(result)
end
