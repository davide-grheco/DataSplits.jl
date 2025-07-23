```@meta
CurrentModule = DataSplits
```

# DataSplits

Documentation for [DataSplits](https://github.com/davide-grheco/DataSplits.jl).

A tiny Julia library for rational train/test splitting algorithms:

| Strategy | Purpose | Complexity |
|----------|---------|------------|
| `KennardStoneSplit` | Maximin split on *X* | `O(N²)` time, `O(N²)` memory |
| `LazyKennardStoneSplit` | Same, streamed | `O(N²)` time, `O(N)` mem |
| `SPXYSplit` | Joint *X–y* maximin (SPXY) | `O(N²)` time, `O(N²)` mem |
| `OptiSimSplit`         | Optimisable dissimilarity-based splitting       | `O(N²)` time, `O(N²)` memory |
| `MinimumDissimilaritySplit`|  Greedy dissimilarity with one candidate | O(N²) time, O(N²) memory |
| `MaximumDissimilaritySplit`|  Greedy dissimilarity with full pool | O(N²) time, O(N²) memory |
| `ClusterShuffleSplit`|  Cluster-based shuffle split | O(N²) time, O(N²) memory |
| `ClusterStratifiedSplit`|  Cluster-based stratified split (equal, proportional, Neyman). Selects a quota per cluster, then splits into train/test according to user fraction. | O(N²) time, O(N²) memory |

All splitting strategies in DataSplits are designed to work with any AbstractArray, including those with non-standard axes. This is achieved by mapping user indices to internal positions, ensuring correctness and extensibility for all data types.

```julia
julia> using DataSplits, Distances

julia> train, test = split(X, KennardStoneSplit(0.8))
julia> train, test = split((X, y), SPXYSplit(0.7; metric = Cityblock()))
julia> train, test = split(X, ClusterStratifiedSplit(clusters, :equal; n=4, frac=0.7))
```
