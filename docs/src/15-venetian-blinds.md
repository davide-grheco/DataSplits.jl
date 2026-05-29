```@meta
CurrentModule = DataSplits
```

# Venetian Blinds CV

**Venetian-blinds cross-validation** sorts samples by target value and assigns
them to folds in a round-robin pattern: the sample ranked 1st goes to fold 1,
ranked 2nd to fold 2, …, ranked k-th to fold k, ranked (k+1)-th back to fold 1,
and so on.

The result resembles the slats of venetian blinds: each fold contains samples
spread evenly across the full sorted range.

## How it works

```text
order = sortperm(target)          # rank samples by target value
for i, idx in enumerate(order):
    fold_test[mod1(i, k)] ← idx  # round-robin assignment
```

Each fold receives roughly N/k samples with a systematic spread of values —
no binning, no randomness (unless `shuffle=true` for tie-breaking).

## VenetianBlindsCV vs. KFold vs. StratifiedKFold

| | [`KFold`](@ref) | [`StratifiedKFold`](@ref) | [`VenetianBlindsCV`](@ref) |
| --- | --- | --- | --- |
| Assignment rule | Contiguous blocks | Round-robin within quantile bins | Round-robin on globally sorted order |
| Binning step | None | Yes (k quantile bins) | None |
| Requires `target =` | No | Yes | Optional (falls back to data) |
| Fold spans full value range | No | Approximately | Yes, by construction |
| Handles continuous targets | n/a | Via quantile bins | Directly |

Use **VenetianBlindsCV** when you have a continuous target and want each fold to
see the full response range without having to choose a number of quantile bins.
Common in NIR spectroscopy where samples are ordered by reference value.

## API reference

- [`VenetianBlindsCV`](@ref)

## References

Naes, T.; Isaksson, T.; Fearn, T.; Davies, T. *A User-Friendly Guide to
Multivariate Calibration and Classification*. NIR Publications, 2002.
