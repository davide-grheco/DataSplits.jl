```@meta
CurrentModule = DataSplits
```

# DataSplits

Documentation for [DataSplits](https://github.com/davide-grheco/DataSplits.jl).

A tiny Julia library for rational train/test splitting algorithms:

| Strategy | Purpose | Complexity |
|----------|---------|------------|
| `KennardStoneSplit` | Maximin split on *X* | `O(N²)` time, `O(N²)` memory |
| `LazyKennardStoneSplit` | Same, streamed | `O(N²)` time, **`O(N)` mem** |
| `SPXYSplit` | Joint *X–y* maximin (SPXY) | `O(N²)` time, `O(N²)` mem |

```julia
julia> using DataSplits, Distances

julia> train, test = split(X, KennardStoneSplit(0.8))
julia> train, test = split((X, y), SPXYSplit(0.7; metric = Cityblock()))
```
