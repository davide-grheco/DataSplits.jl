```@meta
CurrentModule = DataSplits
```

# Distance-Based Splitting

Distance-based strategies select the training set so that it **covers the feature
space as uniformly as possible**. The key insight is that a test point near a
training point is an interpolation problem — easy for the model. A test point far
from all training points is an extrapolation problem — what you actually want to
measure.

These strategies are most valuable for **small-to-medium datasets** (up to a few
thousand samples) where a random split could, by chance, cluster training and test
points together and produce an overly optimistic error estimate.

## The family at a glance

| Strategy | Covers | Peak memory | Speed (N=1000) |
| --- | --- | --- | --- |
| [`KennardStoneSplit`](@ref) | X only | O(N²) — full matrix | 7.3 ms — deterministic |
| [`LazyKennardStoneSplit`](@ref) | X only | O(N) — no full matrix | 43 ms (~6× slower) |
| [`SPXYSplit`](@ref) | X + y | O(N²) | 14 ms — deterministic |
| [`LazySPXYSplit`](@ref) | X + y | O(N) | 445 ms (~32× slower) |
| [`MDKSSplit`](@ref) | X (Mahalanobis) + y | O(N²) | 12 ms — deterministic |
| [`LazyMDKSSplit`](@ref) | X (Mahalanobis) + y | O(N) | 708 ms (~57× slower) |
| [`OptiSimSplit`](@ref) | X | O(N²) | 59 ms — tunable via `max_subsample_size` |
| [`LazyOptiSimSplit`](@ref) | X | O(N) | 304 ms (~5× slower) |
| [`MinimumDissimilaritySplit`](@ref) | X | O(N²) | 11 ms — fastest greedy |
| [`LazyMinimumDissimilaritySplit`](@ref) | X | O(N) | 34 ms (~3× slower) |
| [`MaximumDissimilaritySplit`](@ref) | X | O(N²) | 1.3 s — all candidates |
| [`LazyMaximumDissimilaritySplit`](@ref) | X | O(N) | 14.8 s (~11× slower) |
| [`MoraisLimaMartinSplit`](@ref) | X | O(N²) | 6.7 ms — KS + random swap |

For deep dives see: [Kennard–Stone](08-kennard-stone.md), [SPXY](09-spxy.md),
[OptiSim](10-optisim.md), [Morais–Lima–Martin](11-morais-lima-martin.md).

## When to use which

**Use Kennard–Stone** when you have only features (no target) and you want the
simplest, most established diversity-based split. It is the gold standard for
spectroscopic and tabular data calibration sets.

**Use SPXY or MDKS** when you have both features and a regression target and you
care about covering the response range, not just the feature space. SPXY uses
Euclidean distance for both; MDKS uses Mahalanobis for features (accounting for
correlations) and Euclidean for the target.

**Use OptiSim** when Kennard–Stone is too slow or you want a tunable trade-off
between speed and diversity. `max_subsample_size` controls how many candidates are
evaluated at each step; smaller values are faster but greedier.

**Use MinimumDissimilarity** as the fastest greedy option — it is OptiSim with
`max_subsample_size = 1`.

**Use MaximumDissimilarity** when you want the global maximum spread and can afford
the O(N²) cost per step. Note it greedily includes outliers; remove them first if
that is undesirable.

**Use MoraisLimaMartinSplit** when you want the coverage of Kennard–Stone but with a
random perturbation for ensemble diversity.

**Use the Lazy variants** only when the full N×N distance matrix does not fit in RAM
(roughly N > 5 000 on a 32 GiB machine). They never hold more than a constant number
of distance values in memory at once (O(N) peak), but recompute distances on-the-fly,
making them 3–57× slower. Note that profiling tools report higher *total allocated*
bytes for lazy variants because each on-the-fly distance computation allocates a small
transient object; this does not reflect peak resident memory.

## Minimal example

```julia
using DataSplits

res = partition(X, KennardStoneSplit(); train = 0.8, test = 0.2)
X_train, X_test = splitdata(res, X)
```

All strategies accept a custom distance metric as the first constructor argument:

```julia
using Distances
res = partition(X, KennardStoneSplit(Cityblock()); train = 0.8, test = 0.2)
```

For strategies that also use the target variable, pass `target = y`:

```julia
res = partition(X, SPXYSplit(); target = y, train = 0.8, test = 0.2)
```

See [Getting Started](02-getting-started.md) for cohort sizes, three-cohort splits,
DataFrames, and materialising results.
