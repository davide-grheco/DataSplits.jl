```@meta
CurrentModule = DataSplits
```

# 09. SPXY Split

**Constructor:**

```julia
SPXYSplit(frac::Real; metric::SemiMetric=Euclidean())
```

**Description:**
The SPXY algorithm extends Kennard–Stone by considering both the feature matrix (X) and the target vector (y) when splitting data. It constructs a joint distance matrix as the sum of normalized pairwise distances in X and y, then applies the maximin selection. This ensures the training set is diverse in both predictors and response, which is especially important for regression tasks where the target distribution matters.

**When to use:**

- For regression problems where you want the training set to represent both features and target.
- When target stratification is important.

**When not to use:**

- For classification tasks (unless you encode the target appropriately).
- For very large datasets (O(N²) memory).

**Arguments:**

- `frac`: training fraction.
- `metric`: distance metric for both X and y.

**Usage:**

```julia
using DataSplits, Distances
train, test = split((X, y), SPXYSplit(0.7; metric=Cityblock()))
```

**Pros:**

- Balances representation of predictors and response.
- Useful for regression and continuous targets.

**Cons:**

- May overweight target variation if y is noisy.
- Still requires O(N²) memory and computation.

**Note:** `split(X, strategy)` without y will error.
