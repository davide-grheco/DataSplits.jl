```@meta
CurrentModule = DataSplits
```

# Time-Series Splitting

Time-ordered data requires special care because observations may be dependent and the information available for
prediction can change over time. Random splitting can therefore answer the wrong generalisation question or allow
information from held-out periods to influence model development.

For forecasting or prospective prediction, the evaluation design should reproduce the information available at
prediction time: models are trained on earlier observations and evaluated on later observations. Other time-aware
cross-validation schemes may instead hold out contiguous periods while using observations on both sides of the held-out
block when the objective is to estimate performance across a dependent series rather than simulate future forecasting
[robertsCrossvalidation17](@cite).

DataSplits provides several strategies for these different evaluation settings. The appropriate choice depends on
whether the goal is prospective prediction, evaluation across a dependent series, or protection against overlapping
labels or prediction horizons.

## The atomicity rule

All time-series strategies share one convention: observations with the same timestamp are kept together within a fold.
Train/test boundaries therefore fall between distinct time values rather than splitting observations that belong to the
same time point. This is useful when several observations share a timestamp, for example when multiple instruments,
locations, or experimental units are recorded on the same day.

## TimeSplit — single train/test cutoff

[`TimeSplit`](@ref) creates one train/test partition after ordering observations by time. For prospective evaluation,
the earlier observations are used for training and the later observations for evaluation.

The following example uses daily meteorological measurements from the `airquality` dataset:

```@example time
using DataSplits
using RDatasets
using Dates

air = dataset("datasets", "airquality")

timestamps = Date.(1973, air.Month, air.Day)
X = air[:, [:Wind, :Temp]]

(first(timestamps), last(timestamps), size(X))
```

A forward 80/20 split can be constructed with TimeSplitOldest:

```@example time
res = partition(
    X,
    TimeSplitOldest();
    time = timestamps,
    train = 0.8,
    test = 0.2,
)

(last_train = maximum(timestamps[trainindices(res)]), first_test = minimum(timestamps[testindices(res)]))
```

The evaluation cohort begins strictly after the training cohort. The equivalent explicit ordering is:

```julia
res = partition(
    X,
    TimeSplit(:asc);
    time = timestamps,
    train = 0.8,
    test = 0.2 )
```

[`TimeSplitNewest`](@ref), or `TimeSplit(:desc)`, reverses the direction and places the newest observations in training.
This can be useful for retrospective analyses, but it does not represent prospective forecasting because the evaluation
observations precede the training observations.

Because observations with equal timestamps are kept together, the requested cohort proportions may not always be
achievable exactly.

## TimeSeriesSplit — expanding or rolling window CV

[`TimeSeriesSplit`](@ref) produces `k` folds in chronological order. Fold `i` trains on the observations before chunk
`i+1` and tests on chunk `i+1`. By default the training window grows across folds (expanding window); set
`max_train_size` for a rolling window of fixed length.

```@example time
cvs = partition( X, TimeSeriesSplit(5); time = timestamps)

[
(train_size = length(trainindices(fold)),
    train_end = maximum(timestamps[trainindices(fold)]),
    test_start = minimum(timestamps[testindices(fold)]))
        for fold in folds(cvs)
]
```

Every fold preserves the forward ordering: the latest training timestamp precedes the earliest evaluation timestamp. A
rolling window can instead restrict training to the most recent observations:

```@example time
rolling = partition(X, TimeSeriesSplit(5; max_train_size = 60); time = timestamps)

maximum(length(trainindices(fold)) for fold in folds(rolling))

```

This is useful when very old observations are considered less relevant to the current prediction problem or when a fixed
training-history length better represents deployment.

A `gap` removes observations immediately between the training and evaluation cohorts:

```@example time
gapped = partition(X, TimeSeriesSplit(5; gap = 5); time = timestamps)
length(folds(gapped))
```

A gap can be useful when neighbouring observations are strongly dependent or when feature or target windows overlap
across the train/test boundary.

## BlockedCV — test block surrounded by train

[`BlockedCV`](@ref) divides the data into `k` contiguous blocks [robertsCrossvalidation17, bergmeirUse12](@cite). Each
block takes a turn as the test set while all other blocks — both before and after — form the training set. A symmetric
`gap` is removed on both sides of the test block to buffer against autocorrelation.

```@example time
blocked = partition(X, BlockedCV(5); time = timestamps)
fold = folds(blocked)[3]
train_times = timestamps[trainindices(fold)]
test_times = timestamps[testindices(fold)]
test_start = minimum(test_times)
test_end = maximum(test_times)

( training_before = any(t -> t < test_start, train_times), training_after = any(t -> t > test_end, train_times))
```

A symmetric `gap` can exclude observations immediately surrounding each held-out block:

```julia
blocked_gap = partition(X, BlockedCV(5; gap = 2); time = timestamps)
```

Blocked cross-validation can be appropriate when the objective is to estimate performance across a temporally dependent
series rather than simulate prospective forecasting. Because later observations can contribute to training for earlier
evaluation periods, it should not be used when the deployment task requires strictly forward-in-time prediction.

The gap can reduce dependence across the boundary of the held-out block, but the appropriate gap size depends on the
temporal structure of the application.

## PurgedKFold — purging and embargo

[`PurgedKFold`](@ref) is designed for settings where neighbouring observations can share information because their
features, labels, or prediction horizons extend across time. This problem is particularly common in event-based
financial machine learning, where labels may depend on returns over intervals that overlap neighbouring observations
[lopezdepradoAdvances18](@cite).

Like [`BlockedCV`](@ref), each evaluation fold is a contiguous time block and training data may occur on both sides.
[`PurgedKFold`](@ref) additionally removes observations around the held-out block:

- Purge: `purge` training observations immediately before the evaluation block are removed. This can protect against
  overlap between training labels or prediction horizons and the evaluation period.
- Embargo: `embargo` training observations immediately after the evaluation block are removed. This provides an
  additional buffer against dependence immediately following the held-out period.

For example:

```@example time
purged = partition(X, PurgedKFold(5; purge = 2, embargo = 1); time = timestamps)

length(folds(purged))
```

With no purge or embargo, the split reduces to the corresponding blocked design.

The purge and embargo sizes should reflect the structure of the data-generating process, such as the length of
overlapping target windows or the dependence expected around evaluation boundaries. They should not be chosen to improve
the performance estimate.

## Choosing between the four strategies

| Question                                                            | Answer                    |
| ------------------------------------------------------------------- | ------------------------- |
| One prospective train/test cutoff                                   | [`TimeSplit`](@ref)       |
| Repeated forecasting or walk-forward evaluation                     | [`TimeSeriesSplit`](@ref) |
| Hold out contiguous periods while allowing training on both sides   | [`BlockedCV`](@ref)       |
| Buffer held-out periods against overlapping labels or time horizons | [`PurgedKFold`](@ref)     |
