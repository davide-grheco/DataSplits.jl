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
2. Implement `_split(data, s; rng)` returning `(train, test)`.
3. Use `ValidFraction` for fraction validation.

**Example:**

```julia
struct MySplit <: SplitStrategy; frac; end
function _split(data, s::MySplit; rng)
  # ...
  return train_idx, test_idx
end
```
