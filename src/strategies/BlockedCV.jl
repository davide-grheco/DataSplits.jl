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

function BlockedCV(k::Integer; gap::Integer = 0)
  k >= 2 || throw(SplitParameterError("BlockedCV requires k ≥ 2, got k=$k."))
  gap >= 0 || throw(SplitParameterError("BlockedCV requires gap ≥ 0, got gap=$gap."))
  BlockedCV(Int(k), Int(gap))
end

consumes(::BlockedCV) = (:time,)
fallback_from_data(::BlockedCV) = (:time,)

function _partition(data, alg::BlockedCV; time, kwargs...)
  _blocked_cv_partition(data, alg.k, alg.gap, alg.gap; time = time, name = "BlockedCV")
end
