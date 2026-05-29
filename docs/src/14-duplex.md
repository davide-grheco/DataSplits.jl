```@meta
CurrentModule = DataSplits
```

# Duplex

The **Duplex algorithm** (Snee 1977) builds the training and test sets
**simultaneously** via alternating maximin selection, ensuring both cohorts
independently cover the feature space.

Kennard–Stone selects only the training set greedily; whatever is left over
becomes the test set. Duplex avoids this asymmetry: the test set gets the same
diversity guarantee as the training set.

## How it works

1. Find the globally most distant pair (i, j) in X. Assign i → train, j → test.
2. Maintain two *min-distance vectors*: one tracking the minimum distance from
   each unassigned sample to the **train** set, one to the **test** set.
3. While either cohort is not yet full:
   - Add to train the unassigned sample farthest from the current train set.
   - Add to test the unassigned sample farthest from the current test set.
4. When a sample is claimed by either side, it is immediately excluded from the
   other side's candidate pool (cross-invalidation).

## Duplex vs. Kennard–Stone

| | [`KennardStoneSplit`](@ref) | [`DuplexSplit`](@ref) |
| --- | --- | --- |
| Train coverage | Maximised by construction | Maximised |
| Test coverage | No guarantee (leftover) | Maximised |
| Use when... | Only train diversity matters | Both cohorts must be representative |
| Exact split sizes | Yes | Yes |
| Deterministic | Yes | Yes |

## Eager vs. Lazy

| | [`DuplexSplit`](@ref) | [`LazyDuplexSplit`](@ref) |
| --- | --- | --- |
| Peak memory | O(N²) — full distance matrix | O(N) — no full matrix |
| Speed | Fast (vectorised matrix ops) | ~N× slower (on-the-fly distances) |

Prefer the lazy variant when N > ~5 000 on typical hardware.

## API reference

- [`DuplexSplit`](@ref)
- [`LazyDuplexSplit`](@ref)

## References

Snee, R. D. Validation of Regression Models: Methods and Examples.
*Technometrics* 1977, 19(4), 415–428.
<https://doi.org/10.2307/1267881>.
