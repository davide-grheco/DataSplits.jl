```@meta
CurrentModule = DataSplits
```

# Validation Cohort (`WithValidation`)

## Overview

`WithValidation` is a combinator that turns any pair of train/test
strategies into a train/validation/test split. It applies the first
strategy to separate the test set from the rest of the data, and the
second strategy to the remaining train pool to carve out a validation
cohort. The result is a [`TrainValTestSplit`](@ref).

`WithValidation` does not introduce any new selection logic of its own —
it composes strategies that already exist. Any strategy that returns a
`TrainTestSplit` (which is every built-in strategy other than the
combinators themselves) is a valid building block.

## How it works

1. Run `test_strategy` on the full dataset → train_pool / test indices.
2. Restrict `data` and any auxiliary slot vectors (`target`, `time`,
   `groups`) to `train_pool`.
3. Run `val_strategy` on the restricted data → train / val indices in
   pool-local coordinates.
4. Map the local indices back to the global index space and return a
   [`TrainValTestSplit`](@ref).

## Fraction semantics

Each inner strategy interprets its `frac` field relative to its own input,
consistent with the rest of the package. The total fractions therefore
combine multiplicatively. For instance:

```julia
WithValidation(KennardStoneSplit(0.8), KennardStoneSplit(0.8))
# 80 % train_pool / 20 % test, then 80 % train / 20 % val of the pool
# Globally: 64 % train, 16 % val, 20 % test
```

If you need exact global fractions, do the arithmetic on the inner
fractions yourself. This keeps the contract uniform with single-step
strategies and avoids hidden behaviour.

## Slot composition

The slot interface ([`consumes`](@ref) / [`fallback_from_data`](@ref)) of
`WithValidation` is the union of both inner strategies' slots, so any
combination of `:data`, `:target`, `:time` and `:groups` consumers is
supported transparently:

```julia
# Time-aware test split, target-aware validation split
res = partition(
    X,
    WithValidation(TimeSplitOldest(0.8), TargetPropertyHigh(0.8));
    time = dates,
    target = y,
)
```

The slot keywords are subsetted to `train_pool` before the validation
strategy is invoked, so each inner strategy sees a vector aligned with
its input.

## Usage

```julia
using DataSplits, Distances

# Same strategy for both passes
res = partition(X, WithValidation(KennardStoneSplit(0.8), KennardStoneSplit(0.8)))
X_train, X_val, X_test = splitdata(res, X)

# Different strategies, mixed slots
res = partition(
    X,
    WithValidation(GroupShuffleSplit(0.8), RandomSplit(0.8));
    groups = patient_ids,
)

# SPXY for both — joint X/y diversity preserved in train and val
res = partition(
    X,
    WithValidation(SPXYSplit(0.8), SPXYSplit(0.8));
    target = y,
)
```

## Notes / Limitations

- Both inner strategies must return a [`TrainTestSplit`](@ref). Passing a
  combinator that returns a `TrainValTestSplit` or a
  `CrossValidationSplit` raises `SplitNotImplementedError`.
- The `rng` keyword is forwarded to both inner strategies. If you need
  independent randomness, build a custom strategy or call `partition`
  manually in two stages.
- Group/time/target consistency is the user's responsibility: the slot
  vectors must have one entry per observation in `data`, and the
  validation strategy operates on a contiguous restriction of those
  vectors.

## API Reference

- [`WithValidation`](@ref)
- [`TrainValTestSplit`](@ref)
- [`partition`](@ref)
- [`splitdata`](@ref)
