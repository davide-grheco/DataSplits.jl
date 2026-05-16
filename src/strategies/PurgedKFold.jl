"""
    PurgedKFold(k::Integer; purge::Integer=0, embargo::Integer=0) <: AbstractCVStrategy

Purged k-fold cross-validation for time-dependent data, following the
recipe in López de Prado, *Advances in Financial Machine Learning*
(2018). Observations are sorted by `time=` and partitioned into `k`
contiguous chronological blocks; each block takes a turn as the test
cohort while the train cohort is everything else **minus** an
asymmetric exclusion window:

- `purge` observations are removed from the train cohort immediately
  **before** the test block. This mitigates leakage from samples whose
  labels overlap the test period (e.g. labels built from forward-looking
  returns whose horizon reaches into the test window).
- `embargo` observations are removed from the train cohort immediately
  **after** the test block. This mitigates leakage from serial
  correlation between test-period features and the immediately
  subsequent train samples.

This is the asymmetric counterpart of [`BlockedCV`](@ref) (which uses a
single symmetric `gap` on both sides) and the contiguous-block
counterpart of sklearn's `KFold` adapted for time series.

# Atomicity rule

Observations sharing the same timestamp are never split between train
and test of the same fold — block boundaries fall between distinct
time values, mirroring [`TimeSeriesSplit`](@ref) and [`BlockedCV`](@ref).
`purge` and `embargo` are measured in observations and may trim partial
blocks of equal timestamps from the train side; no row ever leaks into
the test cohort.

# Fields
- `k::Int`: Number of folds (must be ≥ 2 and ≤ number of distinct
  time values).
- `purge::Int`: Number of observations excluded from the train cohort
  immediately before the test block (must be ≥ 0; default `0`).
- `embargo::Int`: Number of observations excluded from the train cohort
  immediately after the test block (must be ≥ 0; default `0`).

# Examples
```julia
# 5-fold purged CV with a 2-observation purge and a 1-observation embargo.
cvs = partition(X, PurgedKFold(5; purge = 2, embargo = 1); time = timestamps)

for (X_train, X_test) in splitview(cvs, X)
    fit!(model, X_train); evaluate(model, X_test)
end

# Single-input shorthand when the timestamps are also the data.
cvs = partition(timestamps, PurgedKFold(4; purge = 1))
```

# References
López de Prado, M. *Advances in Financial Machine Learning*. Wiley,
2018, §7.4 ("Purging and Embargoing").
"""
struct PurgedKFold <: AbstractCVStrategy
  k::Int
  purge::Int
  embargo::Int
end

PurgedKFold(k::Integer; purge::Integer = 0, embargo::Integer = 0) =
  PurgedKFold(Int(k), Int(purge), Int(embargo))

consumes(::PurgedKFold) = (:time,)
fallback_from_data(::PurgedKFold) = (:time,)

function _partition(data, alg::PurgedKFold; time, kwargs...)
  alg.k >= 2 ||
    throw(SplitParameterError("PurgedKFold requires k ≥ 2, got k=$(alg.k)."))
  alg.purge >= 0 || throw(
    SplitParameterError("PurgedKFold requires purge ≥ 0, got purge=$(alg.purge)."),
  )
  alg.embargo >= 0 || throw(
    SplitParameterError(
      "PurgedKFold requires embargo ≥ 0, got embargo=$(alg.embargo).",
    ),
  )

  N = numobs(data)
  sorted_dates, order = groupsortperm(time)
  B = length(sorted_dates)

  alg.k <= B || throw(
    SplitParameterError(
      "PurgedKFold(k=$(alg.k)) requires at least k distinct time values; got $B.",
    ),
  )

  chunk_block_end = distribute_blocks(B, alg.k)
  block_offset = group_offsets(sorted_dates, order, time)

  result = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.k)
  for i = 1:alg.k
    test_block_start = i == 1 ? 1 : chunk_block_end[i-1] + 1
    test_block_end = chunk_block_end[i]

    test_lo = block_offset[test_block_start] + 1
    test_hi = block_offset[test_block_end+1]

    # Asymmetric exclusion: purge before the test block, embargo after.
    train_left_hi = test_lo - 1 - alg.purge
    train_right_lo = test_hi + 1 + alg.embargo

    train_left = train_left_hi >= 1 ? order[1:train_left_hi] : Int[]
    train_right = train_right_lo <= N ? order[train_right_lo:N] : Int[]
    train_idx = vcat(train_left, train_right)

    !isempty(train_idx) || throw(
      SplitParameterError(
        "PurgedKFold: fold $i has empty train cohort (purge=$(alg.purge), embargo=$(alg.embargo) too large for the surrounding blocks).",
      ),
    )

    result[i] = TrainTestSplit(train_idx, order[test_lo:test_hi])
  end
  return CrossValidationSplit(result)
end
