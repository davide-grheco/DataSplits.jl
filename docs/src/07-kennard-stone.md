```@meta
CurrentModule = DataSplits
```

# 07. Kennard–Stone Split

**Constructor:**

```julia
KennardStoneSplit(frac::Real; metric::SemiMetric=Euclidean())
```

**Description:**
The Kennard–Stone algorithm is a deterministic method for selecting a representative training set from a dataset. It works by iteratively choosing the sample that is farthest (in feature space) from all previously selected samples, starting from the most distant pair. This ensures that the training set covers the full range of the feature space, making it especially useful for small or structured datasets where random splits may not be representative.

**When to use:**

- When you want a training set that is diverse and covers the feature space uniformly.
- For small to medium datasets where overfitting is a concern.

**When not to use:**

- For very large datasets (memory intensive).
- When you need to account for the target variable (see SPXY).

**Arguments:**

- `frac`: fraction of data in training set (0 < frac < 1).
- `metric`: distance metric (default Euclidean).

**Usage:**

```julia
using DataSplits, Distances
splitter = KennardStoneSplit(0.8)
train, test = split(X, splitter)
```

**Pros:**

- Deterministic and reproducible.
- Ensures diverse, well-spread training set.

**Cons:**

- Memory-intensive for large datasets (stores full distance matrix).
- Ignores the target variable.

**See Also:** `CADEXSplit` (alias)
