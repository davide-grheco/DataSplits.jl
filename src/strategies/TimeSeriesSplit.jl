"""
    TimeSeriesSplit(k::Integer; gap::Integer=0, max_train_size::Union{Nothing,Integer}=nothing) <: AbstractCVStrategy

Time-aware cross-validation. The temporal sequence is partitioned into
`k + 1` chronological chunks; fold `i` (1 ≤ i ≤ k) tests on chunk `i + 1`
and trains on the observations chronologically preceding it.

By default the train cohort expands across all earlier chunks. Pass
`max_train_size` (in observations) to cap the train cohort, mirroring
scikit-learn's `TimeSeriesSplit`: when set, each fold trains on at most
the most recent `max_train_size` observations before the test chunk.

Train and test cohorts in the same fold are separated by `gap` observations
(useful to avoid leakage between adjacent samples in autocorrelated series).

# Atomicity rule

Observations sharing the same timestamp are never split between train and
test of the same fold — chunk boundaries always fall between distinct time
values, mirroring `TimeSplit`. Chunk sizes are therefore measured in
**distinct time values**, not in observations.

`gap` and `max_train_size` are measured in **observations**, matching
sklearn's contract. When either falls inside a block of equal timestamps,
that block is split on the train side — some rows are kept, the rest
dropped. No row leaks into test (the test cohort still starts at the next
chunk), but the train side is no longer block-aligned.

# Fields
- `k::Int`: Number of folds (must be ≥ 2).
- `gap::Int`: Number of observations skipped from the end of the train
  cohort in each fold (must be ≥ 0; default `0`).
- `max_train_size::Union{Nothing,Int}`: When `nothing` (default), the train
  cohort expands across all earlier chunks. When an `Int ≥ 1`, the train
  cohort is capped to that many observations, taken from the most recent
  end (rolling window).

# Notes
- Requires at least `k + 1` distinct time values.
- A fold whose train cohort would be empty (because `gap` consumes it)
  raises `SplitParameterError`.

# Examples
```julia
# Expanding window (default).
cvs = partition(X, TimeSeriesSplit(5); time = timestamps)

for (X_train, X_test) in splitview(cvs, X)
    fit!(model, X_train); evaluate(model, X_test)
end

# Rolling window: train uses at most the last 100 observations.
cvs = partition(X, TimeSeriesSplit(5; max_train_size = 100); time = timestamps)

# Rolling window with a one-observation gap between train and test.
cvs = partition(X, TimeSeriesSplit(5; gap = 1, max_train_size = 100); time = timestamps)
```
"""
struct TimeSeriesSplit <: AbstractCVStrategy
  k::Int
  gap::Int
  max_train_size::Union{Nothing,Int}
end

TimeSeriesSplit(
  k::Integer;
  gap::Integer = 0,
  max_train_size::Union{Nothing,Integer} = nothing,
) = TimeSeriesSplit(
  Int(k),
  Int(gap),
  max_train_size === nothing ? nothing : Int(max_train_size),
)

consumes(::TimeSeriesSplit) = (:time,)
fallback_from_data(::TimeSeriesSplit) = (:time,)

function _partition(data, alg::TimeSeriesSplit; time, kwargs...)
  alg.k >= 2 ||
    throw(SplitParameterError("TimeSeriesSplit requires k ≥ 2, got k=$(alg.k)."))
  alg.gap >= 0 ||
    throw(SplitParameterError("TimeSeriesSplit requires gap ≥ 0, got gap=$(alg.gap)."))
  if alg.max_train_size !== nothing && alg.max_train_size < 1
    throw(
      SplitParameterError(
        "TimeSeriesSplit requires max_train_size ≥ 1 when set, got $(alg.max_train_size).",
      ),
    )
  end

  N = numobs(data)
  length(time) == N || throw(
    SplitInputError(
      "`time` length ($(length(time))) does not match number of observations ($N).",
    ),
  )

  # Group observations by identical timestamp (atomic time blocks).
  date_to_indices = Dict{eltype(time),Vector{Int}}()
  for (i, t) in enumerate(time)
    push!(get!(date_to_indices, t, Int[]), i)
  end
  sorted_dates = sort!(collect(keys(date_to_indices)))
  B = length(sorted_dates)

  alg.k + 1 <= B || throw(
    SplitParameterError(
      "TimeSeriesSplit(k=$(alg.k)) requires at least k+1 distinct time values; got $B.",
    ),
  )

  # Distribute B blocks across k+1 chunks balancing the remainder across the
  # first `B mod (k+1)` chunks (sklearn's `np.array_split` behaviour). Chunk
  # sizes therefore differ by at most one block, instead of dumping the entire
  # remainder into the last fold.
  n_chunks = alg.k + 1
  base, rem = divrem(B, n_chunks)
  chunk_block_end = Vector{Int}(undef, n_chunks)
  acc = 0
  for c = 1:n_chunks
    acc += base + (c <= rem ? 1 : 0)
    chunk_block_end[c] = acc
  end

  # Build the chronological order vector and the block→position offsets.
  order = Int[]
  block_offset = Vector{Int}(undef, B + 1)
  block_offset[1] = 0
  for (b, d) in enumerate(sorted_dates)
    inds = date_to_indices[d]
    append!(order, inds)
    block_offset[b+1] = block_offset[b] + length(inds)
  end

  folds = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.k)
  for i = 1:alg.k
    test_block_start = chunk_block_end[i] + 1
    test_block_end = chunk_block_end[i+1]
    train_block_end = test_block_start - 1

    test_lo = block_offset[test_block_start] + 1
    test_hi = block_offset[test_block_end+1]
    train_hi = block_offset[train_block_end+1] - alg.gap
    train_lo = 1
    if alg.max_train_size !== nothing
      train_lo = max(train_lo, train_hi - alg.max_train_size + 1)
    end

    train_hi >= train_lo || throw(
      SplitParameterError(
        "TimeSeriesSplit: fold $i has empty train cohort (gap=$(alg.gap) too large for the train chunk).",
      ),
    )

    folds[i] = TrainTestSplit(order[train_lo:train_hi], order[test_lo:test_hi])
  end
  return CrossValidationSplit(folds)
end
