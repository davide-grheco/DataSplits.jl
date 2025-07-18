```@meta
CurrentModule = DataSplits
```

# 08. Lazy Kennard–Stone Split

**Constructor:**

```julia
LazyKennardStoneSplit(frac::Real; metric::SemiMetric=Euclidean())
```

**Description:**
The Lazy Kennard–Stone algorithm is a memory-efficient variant of the classic Kennard–Stone split. Instead of precomputing and storing the full distance matrix, it computes distances on the fly as needed. This makes it suitable for larger datasets where the standard approach would be infeasible due to memory constraints. The selection logic is otherwise identical: iteratively select the sample farthest from those already chosen.

**When to use:**

- When your dataset is too large to fit a full distance matrix in memory.
- When you want deterministic, diverse splits but have limited resources.

**When not to use:**

- For very small datasets (classic Kennard–Stone is faster).
- When you need to account for the target variable (see SPXY).

**Arguments:**

- `frac`: fraction for training set.
- `metric`: distance metric (default Euclidean).

**Usage:**

```julia
using DataSplits
splitter = LazyKennardStoneSplit(0.8)
train, test = split(X, splitter)
```

**Pros:**

- Lower memory footprint than classic Kennard–Stone.
- Still deterministic and diverse.

**Cons:**

- More CPU time due to repeated distance calculations.

**See Also:** `KennardStoneSplit`
