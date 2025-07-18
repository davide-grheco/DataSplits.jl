```@meta
CurrentModule = DataSplits
```

# 14. Cluster Shuffle Split

**Constructor:**

```julia
ClusterShuffleSplit(res::ClusteringResult, frac::Real)
ClusterShuffleSplit(f::Function, frac::Real, data)
```

**Description:**
Cluster Shuffle Split is a group-aware splitting strategy. It either takes a precomputed clustering result or a function to generate one, then shuffles the cluster labels and accumulates whole clusters into the training set until the desired fraction is reached. This approach is ideal for grouped or clustered data where splitting within groups would break structure or introduce leakage.

**When to use:**

- When your data has natural groups or clusters (e.g., patients, molecules, batches).
- When you want to avoid splitting groups across train/test.

**When not to use:**

- When clusters are very imbalanced in size (fraction control is coarse).
- For unstructured data with no meaningful groups.

**Arguments:**

- `res` or `f,data`: clustering result or function.
- `frac`: training fraction.
- `rng`: optional RNG.

**Usage:**

```julia
using DataSplits, Clustering
res = sphere_exclusion(X; radius=0.3)
train, test = split(X, ClusterShuffleSplit(res, 0.8))
```

**Pros:**

- Preserves group integrity.
- Easy to randomize splits.

**Cons:**

- Fraction control is coarse; may overshoot/undershoot target.
- Cluster sizes may vary widely.
