```@meta
CurrentModule = DataSplits
```

# 11. Minimum Dissimilarity Split

**Constructor:**

```julia
MinimumDissimilaritySplit(frac::Real; distance_cutoff=0.35, metric=Euclidean())
```

**Description:**
The Minimum Dissimilarity Split is a greedy, fast variant of OptiSim. At each step, it considers only one candidate sample and adds the one that is most dissimilar to the current training set, repeating until the desired fraction is reached. This approach is much faster than full OptiSim but may not achieve the same level of diversity. It is implemented as an alias for OptiSim with `max_subsample_size=1`.

**When to use:**

- When you need a quick, diversity-aware split for large datasets.
- When computational resources are limited.

**When not to use:**

- When maximum diversity is critical (see Maximum Dissimilarity or OptiSim).

**Arguments:**

- `frac`: fraction for training set.
- `distance_cutoff`: similarity threshold.
- `metric`: distance metric.

**Usage:**

```julia
using DataSplits
train, test = split(X, MinimumDissimilaritySplit(0.8))
```

**Pros:**

- Fast and simple.
- Some diversity in training set.

**Cons:**

- Greedy; may miss optimal diversity.
