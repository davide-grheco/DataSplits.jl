```@meta
CurrentModule = DataSplits
```

# 10. OptiSim Split

**Constructor:**

```julia
OptiSimSplit(frac::Real; max_subsample_size=0, distance_cutoff=0.35, metric=Euclidean())
```

**Description:**
OptiSim is an iterative, optimisable splitting algorithm that aims to maximize the diversity of the training set by minimizing within-set similarity. At each step, it generates a temporary subsample of candidates and selects the one most dissimilar to the current training set, repeating until the desired fraction is reached. The algorithm is flexible and can be tuned for speed or quality by adjusting the subsample size and distance cutoff.

**When to use:**

- When you want a highly diverse training set and can afford extra computation.
- For chemical, biological, or other scientific datasets where diversity is critical.

**When not to use:**

- For very large datasets (computationally intensive).
- When you need a fast, simple split.

**Arguments:**

- `frac`: fraction of training samples.
- `max_subsample_size`: size of candidate pool at each step.
- `distance_cutoff`: threshold for similarity filtering.
- `metric`: distance metric.

**Usage:**

```julia
using DataSplits
train, test = split(X, OptiSimSplit(0.8; max_subsample_size=50))
```

**Pros:**

- Produces high-quality, diverse splits.
- Flexible and tunable.

**Cons:**

- Computationally intensive; sensitive to parameters.
