```@meta
CurrentModule = DataSplits
```

# Combinatorial Purged K-Fold

**Combinatorial Purged K-Fold (CPCV)** (López de Prado 2018) generalises
[`PurgedKFold`](@ref) by exhaustively testing *all combinations* of `n_test_folds`
out of `k` time-ordered blocks rather than testing each block exactly once. This
produces C(k, n_test_folds) train/test pairs and eliminates the path-dependency of
standard walk-forward validation.

## Motivation

Standard walk-forward CV (including [`PurgedKFold`](@ref)) evaluates the model on
each temporal block exactly once, in a fixed order. The performance estimate is
therefore a single path through the data — it depends on the specific temporal
order in which the folds appear. Two practitioners using the same model and the same
data can obtain different estimates simply by choosing different fold boundaries.

CPCV addresses this by averaging over all C(k, n_test_folds) possible test-set
combinations. The result is a distribution of performance estimates (one per
combination) rather than a single number, and the average is path-independent.

## How it works

1. Sort observations by `time=` and assign them to `k` contiguous temporal blocks.
2. Enumerate every combination of `n_test_folds` blocks out of `k` using the
   combinatorial formula C(k, n_test_folds).
3. For each combination:
   - **Test set**: union of the observations in the selected blocks.
   - **Exclusion zone**: `purge` observations immediately before each test block
     and `embargo` observations immediately after each test block are removed from
     train (same semantics as [`PurgedKFold`](@ref)).
   - **Train set**: all observations outside the test set and exclusion zone.
4. Return a [`CrossValidationSplit`](@ref) with one fold per combination.

## CPCV vs. PurgedKFold

| | [`PurgedKFold`](@ref) | [`CombinatorialPurgedKFold`](@ref) |
| --- | --- | --- |
| Number of folds | k | C(k, n_test_folds) |
| Each obs in test | Exactly once | C(k−1, n_test_folds−1) times |
| Path dependency | Yes | No |
| Provides performance distribution | No | Yes |
| `n_test_folds = 1` | Equivalent | — |

Setting `n_test_folds = 1` recovers [`PurgedKFold`](@ref) exactly (same k folds,
same purge and embargo).

## Choosing k and n_test_folds

| Goal | Guidance |
| --- | --- |
| Eliminate path dependency, moderate cost | k=6, n_test_folds=2 → 15 folds |
| Richer performance distribution | k=8, n_test_folds=3 → 56 folds |
| Reproduce PurgedKFold | n_test_folds=1 |
| Largest test sets | n_test_folds close to k/2 |

The number of folds grows combinatorially. For k=10 and n_test_folds=4 you get
C(10,4) = 210 folds — each is fast to evaluate but the total run time is
proportional to the fold count.

## API reference

- [`CombinatorialPurgedKFold`](@ref)
- [`PurgedKFold`](@ref)

## References

López de Prado, M. *Advances in Financial Machine Learning*. Wiley, 2018,
§12 ("Backtesting through Cross-Validation").
