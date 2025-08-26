```@meta
CurrentModule = DataSplits
```

# Algorithms

Each splitter returns two sets of indices `(train, test)`, partitioning samples according to the chosen strategy. **DataSplits expects data matrices to be in the Julia ML convention: columns are samples, rows are features.** Choose based on dataset size, desired diversity, or grouping needs.

For custom data types, implement `Base.length` (number of samples) and `Base.getindex(data, i)` (returning the i-th sample) as described in the [MLUtils documentation](https://juliaml.github.io/MLUtils.jl/stable/api/).

## Kennard–Stone Split

Description: Iteratively selects samples by choosing the point farthest from all previously chosen points in the feature space. This approach ensures the training set uniformly covers the distribution of predictors.

Use Cases: Ideal for small to medium datasets where uniform feature coverage improves model generalization.

Pros:

- Deterministic selection ensures reproducible splits.
- Provides diverse representation of feature space.

Cons:

- Requires storing and computing full distance matrix, which can be memory-intensive.

## Lazy Kennard–Stone Split

Description: A streaming variant of Kennard–Stone that computes distances on the fly, avoiding the full matrix. Suitable when memory is constrained.

Use Cases: Large datasets where memory is a bottleneck.

Pros:

- Reduced memory overhead.

Cons:

- May incur extra computational overhead due to repeated distance computations.

## SPXY Split

Description: Extends Kennard–Stone to both features (X) and target variable (y), selecting samples that maximize combined spread. Ensures training set represents both predictors and response.

Use Cases: Regression tasks where preserving target distribution is critical.

Pros:

- Balances feature and response diversity.

Cons:

- Still requires pairwise distance computations; may overweight target variation.

## LazyMDKSSplit

Description: A memory-efficient, lazy implementation of the Minimum Dissimilarity Kennard–Stone (MDKS) algorithm. Uses Mahalanobis distance for X and Euclidean for y, normalized and summed as in SPXY. Suitable for large datasets where storing the full distance matrix is impractical.

Use Cases: Large regression datasets where both feature and target diversity are important, and memory is a bottleneck.

Pros:

- Memory-efficient: avoids storing the full NxN distance matrix
- Balances feature and response diversity
- Deterministic and reproducible

Cons:

- Slightly higher computational cost due to repeated distance calculations

Example:

```julia
splitter = LazyMDKSSplit(0.7)
result = split((X, y), splitter)
X_train, X_test = splitdata(result, X)
```

## OptiSim Split

Description: Applies iterative swapping of samples between train and test sets to minimize within-set dissimilarity, guided by a dissimilarity measure.

Use Cases: When split quality is paramount and computational resources allow.

Pros:

- Can yield high-quality, low-dissimilarity splits.

Cons:

- Computationally intensive; sensitive to initial split.

## Minimum/Maximum Dissimilarity Split

Description: Greedy cluster-based strategy that adds clusters to the training set based on minimal or maximal dissimilarity criteria until the desired fraction is reached.

Use Cases: When a fast, cluster-aware splitting heuristic is sufficient.

Pros:

- Simpler and faster than full optimization.

Cons:

- Greedy nature may miss globally optimal arrangement; cluster sizes may vary.

## Sphere Exclusion Split

Description: Forms clusters by selecting a sample and excluding all neighbors within a specified radius. Entire clusters are then assigned to train or test to meet the split fraction.

Use Cases: Spatial or similarity-based data where locality grouping matters.

Pros:

- Intuitive radius control over cluster sizes.

Cons:

- Sensitive to radius choice; can produce imbalanced clusters.

## Cluster Shuffle Split

Description: Shuffles cluster labels and sequentially collects whole clusters into the training set until reaching the specified fraction, preserving inherent group structure.

Use Cases: Grouped or clustered data where splitting within groups is undesirable.

Pros:

- Maintains group integrity; introduces randomness.

Cons:

- May overshoot or undershoot desired fraction if cluster sizes vary.

## TargetPropertySplit

Description: Partitions data into train/test sets by sorting samples according to a user-specified property (e.g., a column, a function of the sample, or a target value). The order argument (or alias) controls whether the largest/smallest property values are placed in the training set.

Use Cases: Useful for extrapolation or interpolation splits, e.g., when you want to train on the lowest (or highest) values of a property and test on the rest.

Pros:

- Simple and interpretable
- Works with any property (target, feature, etc.)
- Flexible via property function and order aliases

Cons:

- Not diversity-based; may not represent the full data distribution

## TimeSplit

Description: Splits a 1D array of dates/times into train/test sets, grouping by unique date/time values. No group (samples with the same date) is split between train and test. The actual fraction may be slightly above the requested one, but never below.

Use Cases: Useful for time series or temporal data where you want to avoid splitting samples with the same timestamp.

Pros:

- Respects temporal order
- Never splits samples with the same date/time

Cons:

- Fraction may not be exact due to grouping
