```@meta
CurrentModule = DataSplits
```

# Spectral Split

**Spectral splitting** (Klarner et al. 2024; Ng et al. 2001) partitions samples
into train and test cohorts by clustering on the normalised graph Laplacian of a
pairwise affinity matrix. Because structurally similar samples end up in the same
cluster, and clusters are assigned to either train or test, the two
cohorts are more strongly separated than in random or Kennard–Stone splits.

This creates a harder, more realistic evaluation scenario: the model must generalise
to regions of the feature space it has never seen during training, not just
interpolate between nearby training points.

## How it works

1. **Pairwise distance matrix.** Compute the N×N distance matrix using the chosen
   metric (default: Euclidean).
2. **RBF affinity matrix.** Convert distances to affinities using a Gaussian kernel
   `W[i,j] = exp(−d²/(2σ²))` where σ is the median pairwise distance (the *median
   heuristic*).
3. **Normalised graph Laplacian.** Form `L = I − D^{−1/2} W D^{−1/2}` where
   D is the diagonal degree matrix.
4. **Spectral embedding.** Compute the `n_clusters` smallest eigenvectors of the
   symmetric Laplacian. Row-normalise the resulting N × n_clusters matrix.
5. **K-means clustering.** Run k-means on the embedding (columns = observations).
   This assigns each sample to one of `n_clusters` clusters.
6. **Cluster assignment.** Shuffle the clusters randomly, then add clusters to the
   training cohort until `n_train` is reached; remaining clusters go to test.

## Notes on split sizes

Because clusters are added atomically, the actual train size may differ from
`n_train` by up to one cluster's worth of samples. Pass larger `n_clusters` to reduce the maximum overshoot.

## Tuning n_clusters

`n_clusters` controls the granularity of the partition:

- **Fewer clusters** → larger atomic blocks → sizes may overshoot more, but the
  train/test separation is more pronounced.
- **More clusters** → smaller blocks → sizes are more accurate, but the structural
  separation is weaker (approaches random assignment as n_clusters → N).

A value in the range `[5, 20]` is typically a good starting point.

## API reference

- [`SpectralSplit`](@ref)

## References

Klarner, L. et al. Drug Discovery under Covariate Shift with Domain-Informed Prior Distributions over Functions. arXiv 2023. <https://doi.org/10.48550/ARXIV.2307.15073>.

Ng, A. Y.; Jordan, M. I.; Weiss, Y. On Spectral Clustering: Analysis and an
Algorithm. *Advances in Neural Information Processing Systems* 2001, 14.
