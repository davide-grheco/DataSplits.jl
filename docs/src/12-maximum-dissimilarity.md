```@meta
CurrentModule = DataSplits
```

# 12. Maximum Dissimilarity Split

**Constructor:**

```julia
MaximumDissimilaritySplit(frac::Real; distance_cutoff=0.35, metric=Euclidean())
```

**Description:**
The Maximum Dissimilarity Split is a greedy, diversity-maximizing variant of OptiSim. At each step, it considers all remaining candidate samples and selects the one that is maximally dissimilar (i.e., has the largest minimum distance) to the current training set. This approach is slower than Minimum Dissimilarity but can yield a more diverse training set. It is implemented as an alias for OptiSim with `max_subsample_size=N`.

**When to use:**

- When you want the most diverse training set possible and can afford extra computation.
- For scientific or chemical datasets where outlier coverage is important.

**When not to use:**

- When speed is more important than diversity.
- If you want to avoid including outliers (pre-filter them first).

**Arguments:**

- `frac`: training fraction.
- `distance_cutoff`: similarity threshold.
- `metric`: distance metric.

**Usage:**

```julia
using DataSplits
train, test = split(X, MaximumDissimilaritySplit(0.8))
```

**Pros:**

- Promotes maximum diversity in training set.
- Useful for challenging, representative splits.

**Cons:**

- Higher computational cost.
- May include outliers if present in data.
