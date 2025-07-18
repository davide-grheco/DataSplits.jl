```@meta
CurrentModule = DataSplits
```

# 13. Sphere Exclusion Split

**Constructor:**

```julia
SphereExclusionSplit(frac::Real; radius::Real, metric=Euclidean())
```

**Description:**
Sphere Exclusion Split is a clustering-based splitting strategy. It works by iteratively picking an unassigned sample and forming a cluster of all points within a specified radius (in normalized distance). This process repeats until all samples are assigned to clusters. Clusters are then assigned to the training set until the desired fraction is reached. This method is especially useful for spatial or similarity-based data, where you want to avoid splitting local neighborhoods.

**When to use:**

- For spatial, chemical, or biological data where locality matters.
- When you want to avoid splitting similar samples across train/test.

**When not to use:**

- When you do not know how to set a meaningful radius.
- For data with no meaningful distance metric.

**Arguments:**

- `frac`: training fraction.
- `radius`: normalized distance threshold [0,1].
- `metric`: distance metric.

**Usage:**

```julia
using DataSplits
train, test = split(X, SphereExclusionSplit(0.7; radius=0.2))
```

**Pros:**

- Intuitive spatial grouping.
- Avoids splitting local neighborhoods.

**Cons:**

- Sensitive to radius; cluster sizes may be uneven.
- May not work well for high-dimensional or non-metric data.
