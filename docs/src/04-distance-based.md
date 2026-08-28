```@meta
CurrentModule = DataSplits
```

# Distance-Based Splitting

Distance-based strategies in DataSplits are best understood as sample-subset selection methods. Rather than selecting
training observations according to their frequency in the observed data, they deliberately choose observations that
cover a feature space, a response range, or both.

This approach has a long history in chemometrics, where a large pool of candidate samples may be available, but only a
subset can be included in a calibration experiment or subjected to expensive reference measurements. In this setting,
the aim is to construct a training or calibration set that represents as much of the available domain as possible
[daszykowskiOptimal02](@cite) [bianReview25](@cite).

This is a different objective from estimating future prediction performance. A strategy that constructs a useful
training set does not necessarily leave behind a representative test set.

## What problem do distance-based methods solve?

Random sampling follows the density of the observed data: observations from dense regions are more likely to be selected
than observations from sparse regions. This is appropriate when the aim is to reproduce that distribution, but it can be
undesirable when the objective is to construct a calibration set that covers the available domain with a limited number
of samples.

Distance-based selection instead uses a notion of similarity between observations to favour samples that add new
coverage to the selected set. Depending on the strategy, this similarity may be defined using:

- The predictor or feature space `X`;
- Both predictors `X` and response `y`;
- A covariance-aware distance such as Mahalanobis distance; or
- A stochastic or approximate search over candidate observations.

The result is generally a training set that is more uniformly spread through the space used for selection.

This is appropriate when the aim is to reproduce that distribution, but it can be undesirable when the objective is to
construct a calibration set that covers the available domain with a limited number of samples.

## Training-set design is not performance evaluation

The distinction between selecting informative training observations and selecting representative evaluation observations
is important.

A method such as Kennard–Stone deliberately selects observations near the boundaries and poorly covered regions of the
feature space for training. The observations left over therefore tend to occupy regions already covered by the selected
training set. They are not an independent or random sample from the original population, because their inclusion in the
test cohort is determined by the training-set selection procedure.

Use the complement as an evaluation set only when the question answered by that particular split matches the scientific
or deployment question you intend to study.

> When prospective performance is the objective, reserve evaluation data using a scheme that reflects the intended
> deployment population before applying distance-based sample selection. If performance is estimated by resampling,
> treat the distance-based selection step as part of model development and repeat it using only the training
> observations available in each evaluation fold.

[xuSplitting18](@citet) compared several validation procedures against large independent blind test sets and found that
Kennard–Stone and SPXY could give poor estimates of generalisation performance. In their experiments, the estimates
tended to be pessimistic with very small training sets and optimistic with larger training fractions. However, the
models selected using these methods often performed similarly to those selected by other resampling procedures.

Other studies have likewise shown that reported validation performance depends on how the data are partitioned. In six
QSAR/QSPR case studies, [puzynInvestigating11](@citet) reported better external-validation statistics for splits that
used descriptor information than for splits based only on the response.

These results are not contradictory: the quality of the training set and the quality of the performance estimate are
different properties of a split [crucittiTrain26](@cite).

## The family at a glance

| Strategy                                                                      | Selection space | Main purpose                                                      |
| ----------------------------------------------------------------------------- | --------------- | ----------------------------------------------------------------- |
| [`KennardStoneSplit`](@ref) / [`LazyKennardStoneSplit`](@ref)                 | `X`             | Maximin coverage of feature space                                 |
| [`SPXYSplit`](@ref) / [`LazySPXYSplit`](@ref)                                 | `X` + `y`       | Coverage of predictors and response jointly                       |
| [`MDKSSplit`](@ref) / [`LazyMDKSSplit`](@ref)                                 | `X` + `y`       | Joint coverage using Mahalanobis distance in feature space        |
| [`OptiSimSplit`](@ref) / [`LazyOptiSimSplit`](@ref)                           | `X`             | Approximate space-filling selection with tunable candidate search |
| [`MinimumDissimilaritySplit`](@ref) / [`LazyMinimumDissimilaritySplit`](@ref) | `X`             | Fast greedy dissimilarity-based selection                         |
| [`MaximumDissimilaritySplit`](@ref) / [`LazyMaximumDissimilaritySplit`](@ref) | `X`             | Evaluate the full candidate set when maximizing dissimilarity     |
| [`MoraisLimaMartinSplit`](@ref)                                               | `X`             | Randomised perturbation of a Kennard–Stone selection              |

The individual strategy pages describe their algorithms, assumptions, and original methodological references in more
detail.

## Choosing a strategy

When the response is unavailable or should play no role in sample selection, use an `X`-only strategy such as
[`KennardStoneSplit`](@ref) or [`OptiSimSplit`](@ref).

When the response is already known for the candidate pool and coverage of its range is also important,
[`SPXYSplit`](@ref) or [`MDKSSplit`](@ref) can incorporate that information into selection. Because these methods use
`y` during selection, they are not appropriate for choosing which samples to label when obtaining the reference response
is itself the expensive part of the experiment.

For large candidate pools, prefer a lazy strategy when storing the full pairwise distance matrix becomes prohibitive.
The full implementations cache pairwise distances and are generally faster, while the lazy variants avoid the quadratic
distance matrix by recomputing distances as required.

## Measuring feature-space coverage

A useful way to compare sample-selection strategies is to measure the distance from each observation to its nearest
training observation. The largest of these distances is sometimes called the **fill distance**: smaller values indicate
that the selected training set leaves smaller uncovered regions of the observed feature space.

The following example uses the Boston housing dataset, containing 506 census tracts described by 13 continuous
predictors. Because Euclidean distance is sensitive to feature scale, the predictors are standardised before selection.

```@example distance
using DataSplits
using RDatasets
using Statistics
using Random

boston = dataset("MASS", "Boston")

X = permutedims(Matrix(boston[:, 1:13]))
X = (X .- mean(X; dims = 2)) ./ std(X; dims = 2)

size(X)
```

To make the effect of sample selection easy to see, we deliberately construct a small training subset containing 20% of
the observations, and we calculate the largest distance between any observation and its nearest selected training
observation:

```@example distance
function fill_distance(X, split)
    train = trainindices(split)

    maximum(
        minimum(
            sqrt(sum(abs2, X[:, i] .- X[:, j]))
            for j in train
        )
        for i in axes(X, 2)
    )
end

random_fill = [
    fill_distance(
        X,
        partition(
            X,
            RandomSplit();
            train = 0.2,
            test = 0.8,
            rng = Xoshiro(seed),
        ),
    )
    for seed in 1:100
]

ks_fill = fill_distance(X,
    partition(
        X,
        KennardStoneSplit();
        train = 0.2,
        test = 0.8,
))

(
    kennard_stone = ks_fill,
    random_median = median(random_fill),
    random_range = extrema(random_fill),
)
```

Kennard–Stone is explicitly designed to reduce this kind of uncovered region, so it should provide broader feature-space
coverage than an arbitrary random subset of the same size.

This diagnostic measures **coverage**, not predictive performance. A smaller fill distance does not imply that the
observations left over by the algorithm provide a better estimate of future prediction error.

## The distance metric matters

A distance-based strategy can only be as meaningful as the distance used to define similarity.

Euclidean distance is sensitive to the scale of the input variables. When features use different units or have very
different variances, standardisation or another appropriate transformation should normally be considered before
selection.

High-dimensional, redundant, or strongly correlated feature spaces can also make raw Euclidean distances difficult to
interpret. Depending on the application, a domain-specific metric, dimensionality reduction, or a covariance-aware
distance may better reflect meaningful similarity between observations.

All distance-based DataSplits strategies that support custom metrics accept the metric through their constructor. For
example:

```julia
using Distances

res = partition(
    X,
    KennardStoneSplit(Cityblock());
    train = 0.8,
    test = 0.2,
)
```

The preprocessing and metric should be chosen from scientific knowledge of the problem rather than solely from which
choice produces the largest apparent separation between cohorts.

## Full and lazy implementations

Many distance-based algorithms repeatedly evaluate pairwise distances. The standard implementations can cache an
`N \times N` distance matrix, which gives quadratic memory growth with the number of observations.

The corresponding `Lazy*` strategies avoid storing the full matrix and compute distances as required. This reduces peak
memory requirements at the cost of additional computation.

Prefer the standard implementation when the distance matrix comfortably fits in memory and the lazy implementation when
memory, rather than computation time, is the limiting resource. Exact run times depend strongly on the number of
observations, number of features, distance metric, hardware, and Julia version, so they should be benchmarked for the
dataset of interest rather than inferred from a fixed timing table.

For algorithm-specific details, see [Kennard–Stone Family](08-kennard-stone-family.md).
