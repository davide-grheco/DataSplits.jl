```@meta
CurrentModule = DataSplits
```

# DataSplits

DataSplits is a Julia library for rational train/test splitting algorithms. It provides a variety of strategies for splitting datasets. In several applications random selection is not an appropriate choice and may lead to overestimating model performance.

## Quick Start

```julia
using DataSplits, Distances

# Kennard–Stone split (maximin)
splitter = KennardStoneSplit(0.8)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)

# SPXY split (joint X–y diversity)
splitter = SPXYSplit(0.7; metric=Cityblock())
result = split((X, y), splitter)
X_train, X_test = splitdata(result, X)

# Cluster-based split
using Clustering
clusters = sphere_exclusion(X; radius=0.3)
splitter = ClusterShuffleSplit(clusters, 0.8)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```

## Cheat Sheet

| Task | Strategy | Example |
|------|----------|---------|
| Maximin split | `KennardStoneSplit` | `split(X, KennardStoneSplit(0.8))` |
| Joint X–y split | `SPXYSplit` | `split((X, y), SPXYSplit(0.7))` |
| Cluster shuffle | `ClusterShuffleSplit` | `split(X, ClusterShuffleSplit(clusters, 0.8))` |
| Cluster stratified | `ClusterStratifiedSplit` | `split(X, ClusterStratifiedSplit(clusters, :proportional; frac=0.7))` |
| Time-based split | `TimeSplit` | `split(dates, TimeSplit(0.7))` |
| Property-based split | `TargetPropertySplit` | `split(y, TargetPropertyHigh(0.8))` |
| Random split | `RandomSplit` | `split(X, RandomSplit(0.7))` |
| Randomized Kennard Stone | `MoraisLimaMartinSplit` | `split(X, MoraisLimaMartinSplit(0.8; swap_frac=0.1))` |

## Supported Strategies

| Strategy | Purpose | Complexity |
|----------|---------|------------|
| `KennardStoneSplit` | Maximin split on *X* | `O(N²)` time, `O(N²)` memory |
| `LazyKennardStoneSplit` | Same, streamed | `O(N²)` time, `O(N)` mem |
| `SPXYSplit` | Joint *X–y* maximin (SPXY) | `O(N²)` time, `O(N²)` mem |
| `LazySPXYSplit` | Joint *X–y* maximin (SPXY), streamed | `O(N²)` time, `O(N)` mem |
| `LazyMDKSSplit` | Minimum Dissimilarity Kennard–Stone (MDKS), lazy | `O(N²)` time, `O(N)` mem |
| `MoraisLimaMartinSplit` | Kennard–Stone + random swap | `O(N²)` time, `O(N²)` memory |
| `OptiSimSplit`         | Optimisable dissimilarity-based splitting       | `O(N²)` time, `O(N²)` memory |
| `MinimumDissimilaritySplit`|  Greedy dissimilarity with one candidate | O(N²) time, O(N²) memory |
| `MaximumDissimilaritySplit`|  Greedy dissimilarity with full pool | O(N²) time, O(N²) memory |
| `ClusterShuffleSplit`|  Cluster-based shuffle split | O(N²) time, O(N²) memory |
| `ClusterStratifiedSplit`|  Cluster-based stratified split (equal, proportional, Neyman). Selects a quota per cluster, then splits into train/test according to user fraction. | O(N²) time, O(N²) memory |

All splitting strategies in DataSplits are designed to work with any AbstractArray, including those with non-standard axes.

**DataSplits expects data matrices to follow the Julia ML convention: columns are samples, rows are features.** If your data uses rows as samples, transpose it before splitting (e.g., use `X'`).

For custom data types, implement `Base.length` (number of samples) and `Base.getindex(data, i)` (returning the i-th sample) as described in the [MLUtils documentation](https://juliaml.github.io/MLUtils.jl/stable/api/). This ensures compatibility with all DataSplits algorithms and utilities.
