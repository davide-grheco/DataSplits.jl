```@meta
CurrentModule = DataSplits
```

# 05. Extending DataSplits

## Custom Clustering

To add a new clustering algorithm:

1. Subtype `ClusteringResult`.
2. Implement `assignments`, `nclusters`, `counts`, `wcounts` for your result type.
3. Register a clustering function returning your result.

**Example:**

```julia
struct MyClusteringResult <: ClusteringResult
  assignments::Vector{Int}
end
assignments(r::MyClusteringResult) = r.assignments
nclusters(r::MyClusteringResult) = maximum(r.assignments)
# ...
```

## Custom Splitter

To add a new splitting strategy:

1. Subtype `SplitStrategy`.
2. Implement `_split(data, s; rng)` to call.
3. Use `ValidFraction` for fraction validation.

**Example:**

```julia
struct MySplit <: SplitStrategy; frac; end

function mysplit(N, s, rng, data)
    cut = floor(Int, s.frac * N)
    train_pos = 1:cut
    test_pos = (cut+1):N
    return train_pos, test_pos
end

function _split(data, s::MySplit; rng=Random.default_rng())
    N = numobs(data)
    train_pos, test_pos = mysplit(N, s, rng, data)
    return TrainTestSplit(train_pos, test_pos)
end
```
