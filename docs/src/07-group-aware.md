```@meta
CurrentModule = DataSplits
```

# Group-Aware Splits

Group labels can play different roles when constructing training and evaluation cohorts. DataSplits supports two
distinct designs:

- Hold out whole groups when the goal is to evaluate generalisation to groups not represented during training.
- Sample within groups when the goal is to preserve representation of each group in both cohorts.

These designs answer different scientific questions and should not be used interchangeably.

For example, repeated measurements from the same patient should usually be kept together when evaluating performance on
new patients. In contrast, if future predictions patients already represented during training, splitting observations
within each patient may be appropriate. The correct design therefore depends on the intended deployment population
rather than on the presence of groups alone [mayData10, robertsCrossvalidation17](@cite).

Groups are supplied through the `groups` keyword of partition. Group labels may be integers, strings, symbols, cluster
assignments, subject identifiers, batch labels, or any other vector defining membership.

## Holding out whole groups

[`GroupShuffleSplit`](@ref) assigns each group entirely to either training or evaluation. No group can therefore appear
in both cohorts.

This is appropriate when performance should be estimated on previously unseen groups, for example:

- New patients when several measurements are available per patient
- New experimental batches or acquisition sites
- New molecular scaffolds
- New spatial or hierarchical units

Randomly splitting individual observations in these settings can place closely related observations in both cohorts and
produce an optimistic performance estimate.

The sleepstudy dataset provides a simple example. It contains repeated observations from 18 subjects:

```@example group
using DataSplits
using RDatasets
using Random

sleep = dataset("lme4", "sleepstudy")
X = sleep[:, [:Days]]
subjects = sleep.Subject

(observations = length(subjects), subjects = length(unique(subjects)))
```

A group-aware split keeps subjects completely separate:

```@example group
res = partition(X, GroupShuffleSplit(); groups = subjects, train = 0.8, test = 0.2, rng = Xoshiro(42))
train_subjects = Set(subjects[trainindices(res)])
test_subjects = Set(subjects[testindices(res)])

(train_subjects = length(train_subjects),
test_subjects = length(test_subjects),
shared_subjects = length(intersect(train_subjects, test_subjects)))
```

Because groups are indivisible, the requested cohort proportions may not be achievable exactly.
[`GroupShuffleSplit`](@ref) adds groups to the training cohort in random order until the requested training size is
reached, so the resulting training cohort may be larger than requested.

When the group labels themselves are the data being partitioned, the `groups` keyword may be omitted:

```@example group
group_res = partition(subjects, GroupShuffleSplit(); train = 0.8, test = 0.2, rng = Xoshiro(42))

length(intersect(Set(subjects[trainindices(group_res)]), Set(subjects[testindices(group_res)])))
```

For repeated group-aware evaluation rather than a single train/test split, see [`GroupKFold`](@ref),
[`GroupShuffleSplitCV`](@ref), and [`LeaveOneGroupOut`](@ref) on the Cross-Validation page.

## Sampling within groups

[`GroupStratifiedSplit`](@ref) answers a different question. Instead of keeping groups separate, it samples observations
within each group, so that the selected groups can contribute observations to both training and evaluation cohorts.

This is useful when the groups act as sampling strata whose representation should be preserved across cohorts. It should
not be used to evaluate generalisation to unseen groups, because group membership is intentionally shared between
training and evaluation data.

For example, applying proportional group stratification to the same subject-level data produces a different evaluation
design:

```@example group
stratified = partition(X, GroupStratifiedSplit(:proportional); groups = subjects, train = 0.8, test = 0.2, rng = Xoshiro(42))
train_subjects = Set(subjects[trainindices(stratified)])
test_subjects = Set(subjects[testindices(stratified)])

(train_subjects = length(train_subjects),
test_subjects = length(test_subjects),
shared_subjects = length(intersect(train_subjects, test_subjects)))
```

Here, subjects are represented in both cohorts. This may be appropriate if the intended prediction task concerns new
observations from already represented subjects, but it would not provide an independent assessment of performance on new
subjects.

### Allocation methods

[`GroupStratifiedSplit`](@ref) supports three ways of determining how many observations from each group participate in
the split:

| Allocation      | Behaviour                                                                            | Requires `n` |
| --------------- | ------------------------------------------------------------------------------------ | ------------ |
| `:proportional` | All samples from each group are used.                                                | no           |
| `:equal`        | Pick `n` samples from each group; the rest are excluded from both cohorts.           | yes          |
| `:neyman`       | Pick a quota per group proportional to group size × within-group standard deviation. | yes          |

## Notes and limitations

Whole-group splitting can give cohort sizes that differ from the requested proportions, particularly when groups are
large or strongly imbalanced. If only a small number of groups are available, performance estimates based on a single
held-out group partition may also be unstable; group-aware cross-validation can be preferable when repeated evaluation
is feasible.

- Fraction control is coarse with `GroupShuffleSplit` — actual sizes depend on group sizes and may overshoot the
  requested counts.
- Very imbalanced groups can make stratified allocations degenerate (e.g. a tiny group with `:equal` allocation will be
  fully consumed).
- `:neyman` requires per-feature standard deviations to be finite and non-zero in aggregate; if every group has zero
  within-group variance the allocation falls back to `:equal`.
