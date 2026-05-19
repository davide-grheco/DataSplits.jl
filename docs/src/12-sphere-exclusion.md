```@meta
CurrentModule = DataSplits
```

# Sphere Exclusion

`sphere_exclusion` clusters samples by picking a centre point and assigning to it
all samples within a specified radius. The radius is applied to the
**normalised** distance matrix (scaled to [0, 1]), so it is unitless and portable
across datasets with different absolute scales.

This is a clustering utility, not a split strategy itself. Use the resulting
cluster assignments as the `groups=` argument to any group-aware split or CV
strategy.

## How it works

1. Normalise the pairwise distance matrix to [0, 1].
2. Take the first unassigned sample as a new cluster centre.
3. Assign to that cluster every unassigned sample within `radius` of the centre.
4. Repeat until all samples are assigned.

The number of clusters is determined automatically by `radius` — smaller radii
produce more, smaller clusters; larger radii produce fewer, larger ones.

## Usage

```julia
using DataSplits

result = sphere_exclusion(X; radius = 0.3)
cluster_ids = result.assignments   # Vector{Int}, one entry per sample

# Use assignments as groups for a group-aware split.
res = partition(X, GroupShuffleSplit();
                groups = cluster_ids, train = 0.8, test = 0.2)

# Use with GroupKFold for cluster-stratified cross-validation.
cvs = partition(X, GroupKFold(5); groups = cluster_ids)

# Custom metric.
using Distances
result = sphere_exclusion(X; radius = 0.2, metric = Cityblock())
```

## Choosing the radius

The `radius` parameter (in [0, 1] after normalisation) controls cluster granularity:

| Radius | Effect |
| --- | --- |
| Small (0.05–0.15) | Many small, tight clusters |
| Medium (0.2–0.4) | Moderate cluster sizes — good default |
| Large (0.5+) | Few large clusters; may produce very uneven groups |

There is no single best radius — it depends on the density of your data. A useful
heuristic: choose a radius such that the number of clusters is roughly equal to your
intended number of folds or groups.

## Return value

`sphere_exclusion` returns a [`SphereExclusionResult`](@ref) with:

- `assignments::Vector{Int}` — cluster ID per sample (1-based).
- `radius::Float64` — the radius that was used.
- `metric::Distances.SemiMetric` — the metric that was used.

The result implements the `Clustering.ClusteringResult` interface, so standard
accessor functions (`assignments`, `nclusters`, `counts`) work.

## API reference

- [`sphere_exclusion`](@ref)
- [`SphereExclusionResult`](@ref)
