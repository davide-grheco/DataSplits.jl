```@meta
CurrentModule = DataSplits
```

# Time-Series Splitting

Time-ordered data breaks the i.i.d. assumption underlying standard cross-validation.
The fundamental rule is simple: **always train on the past, evaluate on the future**.
Any split that allows information from a future time point to influence training is
a data leak that will inflate reported performance.

DataSplits provides four strategies for time-ordered data. They differ in how
strictly they enforce temporal order and how they handle the label-overlap leakage
that arises in financial and event-based data.

## The atomicity rule

All time-series strategies share one convention: observations sharing the same
timestamp are never split between train and test in the same fold. Block and chunk
boundaries always fall between distinct time values. This prevents partial-group
leakage when multiple observations carry the same timestamp (e.g. daily or weekly
data with multiple instruments).

## TimeSplit — single train/test cutoff

[`TimeSplit`](@ref) is the simplest strategy: sort observations chronologically and
put the first `train` observations into the training set, the rest into test.

```julia
using DataSplits

# Oldest 80% of observations go to train.
res = partition(X, TimeSplit(:asc); time = timestamps, train = 0.8, test = 0.2)
X_train, X_test = splitdata(res, X)

# Convenience aliases.
res = partition(X, TimeSplitOldest(); time = timestamps, train = 0.8, test = 0.2)
res = partition(X, TimeSplitNewest(); time = timestamps, train = 0.2, test = 0.8)

# When the timestamps vector is the data.
res = partition(timestamps, TimeSplitOldest(); train = 0.8, test = 0.2)
```

`TimeSplit(:asc)` places the oldest observations in train; `TimeSplit(:desc)` places
the newest in train (useful for testing on historical data).

The actual train fraction may slightly overshoot `train` because complete timestamp
groups are never broken.

## TimeSeriesSplit — expanding or rolling window CV

[`TimeSeriesSplit`](@ref) produces `k` folds in chronological order. Fold `i` trains
on the observations before chunk `i+1` and tests on chunk `i+1`. By default the
training window grows across folds (expanding window); set `max_train_size` for a
rolling window of fixed length.

```julia
using DataSplits

# Expanding window: each successive fold trains on more data.
cvs = partition(X, TimeSeriesSplit(5); time = timestamps)

for (X_tr, X_te) in splitview(cvs, X)
    fit!(model, X_tr)
    evaluate(model, X_te)
end

# Rolling window: train on at most the last 200 observations.
cvs = partition(X, TimeSeriesSplit(5; max_train_size = 200); time = timestamps)

# Add a gap between the end of train and the start of test to avoid leakage
# from features that look ahead (e.g. moving averages).
cvs = partition(X, TimeSeriesSplit(5; gap = 5); time = timestamps)
```

Use `TimeSeriesSplit` when you have enough data for expanding or rolling evaluation
and you want a standard walk-forward backtesting setup.

## BlockedCV — test block surrounded by train

[`BlockedCV`](@ref) (Bergmeir & Benítez 2012, Roberts et al. 2017) divides the
data into `k` contiguous blocks. Each block takes a turn as the test set while
**all other blocks** — both before and after — form the training set. A symmetric
`gap` is removed on both sides of the test block to buffer against autocorrelation.

```julia
cvs = partition(X, BlockedCV(5); time = timestamps)

# Remove 2 observations on each side of the test block.
cvs = partition(X, BlockedCV(5; gap = 2); time = timestamps)
```

This differs from [`TimeSeriesSplit`](@ref) in a key way: later blocks can appear
in the training set of earlier test folds. This is valid when the data is
stationary and what you want is average-in-time performance rather than
forward-only prediction.

## PurgedKFold — purging and embargo

[`PurgedKFold`](@ref) is the standard cross-validation strategy for financial machine
learning (López de Prado 2018). It extends blocked CV with an **asymmetric** gap:

- **Purge** (`purge` observations): removed from train *before* the test block, to
  prevent leakage from samples whose labels overlap the test period (e.g. returns
  computed from forward-looking windows that reach into the test window).
- **Embargo** (`embargo` observations): removed from train *after* the test block,
  to prevent leakage from serial correlation between test-period features and the
  immediately subsequent train samples.

```julia
# 5-fold purged CV with a 2-observation purge and 1-observation embargo.
cvs = partition(X, PurgedKFold(5; purge = 2, embargo = 1); time = timestamps)

for (X_tr, X_te) in splitview(cvs, X)
    fit!(model, X_tr); evaluate(model, X_te)
end

# Minimal version — no purge or embargo (equivalent to BlockedCV).
cvs = partition(X, PurgedKFold(5); time = timestamps)
```

## Choosing between the four strategies

| Question | Answer |
| --- | --- |
| Simple cutoff, one split | [`TimeSplit`](@ref) |
| Walk-forward CV, stationary data | [`TimeSeriesSplit`](@ref) |
| Stationary data; train includes blocks before and after test | [`BlockedCV`](@ref) |
| Label overlap leakage (financial returns) | [`PurgedKFold`](@ref) |

## References

- Bergmeir, C.; Benítez, J. M. On the use of cross-validation for time series
  predictor evaluation. *Inf. Sci.* 2012, 191, 192–213.
- Roberts, D. R. et al. Cross-validation strategies for data with temporal, spatial,
  hierarchical, or phylogenetic structure. *Ecography* 2017, 40(8), 913–929.
- López de Prado, M. *Advances in Financial Machine Learning*. Wiley, 2018, §7.4.
