```@meta
CurrentModule = DataSplits
```

# Onion and XYOnion

The **Onion** family of algorithms splits a dataset into training and test sets by
peeling concentric shells — like the layers of an onion — from the data cloud.  The
outer layers, which contain the most "extreme" samples furthest from the centroid, go
to the training set; the next-outermost samples form the test set; the remaining core
is assigned randomly in the requested proportion.

There are two variants:

- [`OnionSplit`](@ref) — uses only `X` (feature matrix). The training set covers the
  boundary of the data space; the test set sits just inside it.
- [`XYOnionSplit`](@ref) — uses both `X` and `y` (target). Distance from the centroid
  is measured jointly in X–y space (the same normalised SPXY distance used by
  [`SPXYSplit`](@ref)), so the outer layers cover both feature and response diversity.

Both share the same layered loop and accept an optional Mahalanobis metric.

## How it works

For a dataset with N samples and a requested `fraction` for training:

1. In each of `n_layers` iterations, select roughly `10 % × fraction × remaining`
   samples as the **outermost** (farthest from the centroid) and assign them to
   **train**; then select the next `10 % × (1−fraction) × remaining` samples and
   assign them to **test**.
2. Repeat until all layers are peeled.
3. Assign the remaining **core** samples randomly in the requested proportion.

The outer-layer selection uses the *distslct* algorithm (Gallagher et al. 2003):

- When the number of samples to select is **≤ F** (number of features): use
  iterative orthogonalisation.  Pick the sample with the largest distance from the
  centroid, project it out of the working matrix (Gram–Schmidt step), and repeat.
- When the number to select **> F**: bootstrap F samples with the orthogonalisation
  method, then extend greedily by cumulative SPXY-style distance — the next sample
  maximises the sum of distances to all already-selected samples.

### Approximate split sizes

Because counts are computed with `round` in each layer, the final number of training
and test samples may differ from the exact requested values by a few samples.  This is
by design — the layered structure takes priority over exact counts.

## Onion vs. XYOnion

| | [`OnionSplit`](@ref) | [`XYOnionSplit`](@ref) |
| --- | --- | --- |
| Requires `target =` | No | Yes |
| Distance space | X only | X + y (normalised SPXY) |
| Good for | Unsupervised coverage | Regression with known y |
| Same as... | XYOnion with y = 0 | — |

Use **`OnionSplit`** when you do not have a target variable, or when covering the
feature space is the primary concern (e.g. spectroscopic calibration, geospatial
sampling, molecular diversity).

Use **`XYOnionSplit`** when you have a regression target and want the outer training
layers to cover the response range as well as the feature space — similar reasoning to
preferring SPXY over Kennard–Stone.

## Onion vs. Kennard–Stone

Both algorithms select training samples that cover the boundary of the data space, but
they do it differently:

- **Kennard–Stone** is a global greedy algorithm: every new sample maximises the
  minimum distance to all previously selected samples.  It guarantees exact split
  sizes and is fully deterministic.
- **Onion** is a layered algorithm: it peels shells of a fixed thickness (10 % of the
  remaining pool) and assigns the remainder randomly.  This produces a more balanced
  distribution across the data space (training samples are not concentrated only at
  the extremes) at the cost of approximate cohort sizes and one random step.

For large `n_layers` the onion layers become thin and the result approaches
Kennard–Stone.  For `n_layers = 1` almost all samples go through the random step.

## Usage

### OnionSplit — X only

```julia
using DataSplits

res = partition(X, OnionSplit(); train = 70, test = 30)
X_train, X_test = splitdata(res, X)
```

### XYOnionSplit — X and y

```julia
using DataSplits

res = partition(X, XYOnionSplit(); target = y, train = 70, test = 30)
X_train, X_test = splitdata(res, X)
y_train, y_test = splitdata(res, y)
```

### Mahalanobis distance

Pass `metric_X = nothing` to compute Mahalanobis distance from the sample covariance
at split time.  This accounts for feature correlations and is recommended when
variables have very different variances or are highly collinear.

```julia
res = partition(X, OnionSplit(; metric_X = nothing); train = 70, test = 30)
res = partition(X, XYOnionSplit(; metric_X = nothing); target = y, train = 70, test = 30)
```

### Controlling the number of layers

The default of 3 layers works well for most datasets.  For small datasets (N < 30)
consider increasing `n_layers` so that each layer contains at least a few samples:

```julia
res = partition(X, OnionSplit(; n_layers = 5); train = 70, test = 30)
```

## API reference

- [`OnionSplit`](@ref)
- [`XYOnionSplit`](@ref)

## References

Ezenarro, J. et al. XYOnion: A Layer-Based Method for Splitting Datasets into
Calibration and Validation Subsets. *Analytica Chimica Acta* 2025, 344229.
<https://doi.org/10.1016/j.aca.2025.344229>.

Gallagher, N.B.; O'Sullivan, D. *Selection of Representative Learning and Test Sets
Using the Onion Method.* Eigenvector Research Technical Report (2022).
<https://eigenvector.com/wp-content/uploads/2022/10/Onion_SampleSelection.pdf>.

Gallagher, N.B.; Shaver, J.M.; Martin, E.B.; Morris, J.; Wise, B.M.; Windig, W.
Curve resolution for images with applications to TOF-SIMS and Raman.
*Chemometrics and Intelligent Laboratory Systems* 2003, 77(1), 105–117.
