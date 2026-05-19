```@meta
CurrentModule = DataSplits
```

# Group-Aware Splits

Group-aware strategies keep entire groups together when partitioning. This is
the right choice when observations within a group are not independent —
e.g. multiple measurements per patient, molecules sharing a scaffold,
samples from the same batch, points within a spatial cluster — and splitting
a group across train and test would leak information.

Groups are passed as a vector of membership IDs via the `groups=` keyword of
[`partition`](@ref). Any labelling is valid: integer cluster assignments,
patient IDs, strings, symbols, etc. When `data` itself is the group vector,
the `groups=` keyword may be omitted (slot fallback).

DataSplits provides two group-aware strategies:

- [`GroupShuffleSplit`](@ref) — shuffle and accumulate whole groups until the
  requested train size is reached.
- [`GroupStratifiedSplit`](@ref) — split each group internally so that both
  cohorts contain samples from every group, with several allocation methods.

## When to use

- Data has natural groups (patients, molecules, batches, sites, clusters).
- Within-group observations are correlated and a leaky split would inflate
  test performance.
- You have a clustering result (e.g. from `sphere_exclusion`, `kmeans`, …)
  and want to use the assignments as groups.

## When not to use

- Observations are independent — a non-group strategy is simpler.
- A single group dominates the dataset — group-aware splits become trivial
  or degenerate.

## `GroupShuffleSplit`

Adds entire groups (in random order) to the train cohort until the requested
training size is reached; remaining groups go to test. Because groups are
added whole, the actual train size may overshoot `train`. No attempt is made
to minimise the overshoot.

```julia
using DataSplits

# groups supplied alongside the data
res = partition(X, GroupShuffleSplit(); groups=patient_ids, train=80, test=20)
X_train, X_test = splitdata(res, X)

# data itself is the group vector — `groups=` may be omitted
res = partition(patient_ids, GroupShuffleSplit(); train=0.8, test=0.2)

# clustering result as groups
using Clustering
res = partition(X, GroupShuffleSplit();
                groups=assignments(kmeans(X, 5)), train=80, test=20)
```

## `GroupStratifiedSplit`

Splits *within* each group so that both train and test contain samples from
every group. Useful when you want each group represented in both cohorts.

The within-group train fraction is derived from the global cohort sizes
(`n_train / N`). Three allocation methods control how many samples per group
are eligible for the split:

| Allocation       | Behaviour                                                                                  | Requires `n` |
|------------------|--------------------------------------------------------------------------------------------|--------------|
| `:proportional`  | All samples from each group are used.                                                      | no           |
| `:equal`         | Pick `n` samples from each group; the rest are discarded.                                  | yes          |
| `:neyman`        | Pick a quota per group proportional to group size × within-group standard deviation.       | yes          |

```julia
using DataSplits

# Proportional — every sample participates.
res = partition(X, GroupStratifiedSplit(:proportional);
                groups=patient_ids, train=80, test=20)

# Equal — 5 samples per group, then split.
res = partition(X, GroupStratifiedSplit(:equal; n=5);
                groups=patient_ids, train=80, test=20)

# Neyman — quota weighted by within-group dispersion.
res = partition(X, GroupStratifiedSplit(:neyman; n=10);
                groups=patient_ids, train=80, test=20)
```

## Notes and limitations

- Fraction control is coarse with `GroupShuffleSplit` — actual sizes depend
  on group sizes and may overshoot the requested counts.
- Very imbalanced groups can make stratified allocations degenerate (e.g. a
  tiny group with `:equal` allocation will be fully consumed).
- `:neyman` requires per-feature standard deviations to be finite and
  non-zero in aggregate; if every group has zero within-group variance the
  allocation falls back to `:equal`.

## API reference

- [`GroupShuffleSplit`](@ref)
- [`GroupStratifiedSplit`](@ref)

## References

May, R. J.; Maier, H. R.; Dandy, G. C. *Data Splitting for Artificial Neural
Networks Using SOM-Based Stratified Sampling.* Neural Networks 2010, 23(2),
283–294.
