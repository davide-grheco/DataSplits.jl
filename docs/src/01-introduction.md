```@meta
CurrentModule = DataSplits
```

# Why Splitting Matters

Data splitting is one of the first and most important decisions in model development. Before fitting a model, you need
to decide which data you will use, how to select an appropriate model, and how to evaluate if the model is useful.

A typical modelling workflow separates three stages [pizarroMultiple02](@cite):

- Model fitting (training): estimate model parameters and fit any data-dependent preprocessing using the training data.
- Model selection: choose among candidate models, features, and hyperparameters using validation data or
  cross-validation.
- Performance evaluation: estimate the generalisation performance of the selected modelling procedure using test data
  that played no role in fitting or model selection.

Performance evaluation before deployment is not the only way to assess a model. Once deployed, its performance should
also be monitored on new data to detect changes in the data distribution or degradation over time. Pre-deployment
evaluation serves a different purpose: it provides evidence that the modelling procedure is likely to meet the required
level of performance before it is put into use. This is particularly important when poor predictions can have
substantial financial, operational, or safety consequences, for example when predictions inform infrastructure
investments or clinical treatment decisions.

The way you split the data determines what that final estimate means. A random split estimates performance on new
observations drawn similarly to the current sample; a group-aware split tests transfer to unseen groups; a temporal
split tests prediction of future observations; and an extrapolation split deliberately tests performance outside the
training data.

There is no universally correct way to split data, because splitting serves two related but distinct purposes. First, it
determines which observations are available for fitting, and therefore can affect the quality of the model that is
trained. Second, it determines which observations are used for validation and testing, and therefore can affect the
estimated generalisation performance. A strategy that produces a useful training set is not necessarily a good strategy
for estimating prospective performance, and vice versa.

After model selection and final evaluation, it is common to refit the selected model on all available data before
deployment. This allows the final model to benefit from all available observations. The evaluation data should not,
however, be used to make further modelling choices: the reported evaluation remains an estimate obtained before this
final refitting step.

The simplest and most widely used approach is a random split, typically into a training set and a test set. When
observations are independent and future observations follow the same distribution as the available data, random
splitting provides a natural estimate of generalisation performance. Its limitations become important when the dataset
is small or when observations have additional structure, such as groups, temporal ordering, or strong similarity
relationships.

## Small datasets

With a small dataset, a single train/test split can be unstable: both the model that is fitted, and the performance
measured on the test set can depend strongly on which observations happen to fall into each cohort. Cross-validation or
repeated resampling can reduce the dependence of the estimate on a single random partition.

Suppose you have 100 chemical compounds and randomly assign 80 to training and 20 to test. Random sampling represents
dense regions of the observed feature distribution more often than sparse ones. With a small dataset, this can leave
parts of the feature space poorly represented in the training set.

Whether that matters depends on the goal. If future compounds are drawn from the same distribution as the observed
sample, a random test set estimates performance under that distribution. If instead the aim is to construct a
calibration set that spans the available space, coverage becomes part of the design problem.

Diversity-based strategies such as Kennard–Stone, SPXY, and OptiSim select training samples to spread across the
available domain. They can reduce gaps in training-set coverage, but this is a different objective from constructing a
representative random test set: the observations left over by a diversity algorithm are determined by the design and
should not automatically be interpreted as an unbiased sample of future data.

## Group leakage

Many datasets contain natural groups: repeated measurements from the same patient, assays from the same batch, molecules
sharing the same scaffold, or observations from the same geographic site. Observations within a group are often more
similar to one another than to observations from other groups because they share characteristics, experimental
conditions, or other sources of dependence.

If a random split places some observations from patient 17 in the training set and others in the test set, the model may
exploit patient-specific information that it has effectively seen before. The resulting test performance can then
overestimate how well the model will generalise to entirely new patients. This is **group leakage**.

Whether this matters depends on the intended use of the model. If deployment involves predictions for previously unseen
groups, evaluation should hold out entire groups. If instead the model will make new predictions for groups already
represented during training, holding out whole groups answers a different, usually more demanding, generalisation
question.

A **group-aware split** ([`GroupShuffleSplit`](@ref), [`GroupKFold`](@ref)) keeps all observations from the same group
within the same cohort, preventing information about a group from being shared across training and evaluation data.

## Temporal leakage

Time-ordered data requires evaluation to respect the information that would have been available when each prediction was
made. A random split can violate this ordering by placing earlier observations in the test set and later observations in
the training set, allowing the model to benefit from information that would not have been available in practice.

For forecasting problems, the evaluation procedure should therefore reproduce the direction of deployment: models are
trained on past observations and evaluated on later ones. This restriction should also apply to data-dependent
preprocessing and feature construction, which must be fitted using training data only.

A **time-aware split** ([`TimeSeriesSplit`](@ref), [`BlockedCV`](@ref), [`TimeSplit`](@ref)) preserves temporal order by
training on the past and evaluating on the future.

## Extrapolation

Sometimes the scientific question itself requires extrapolation. If you are building a model to predict observations
outside the training domain, the evaluation set should reproduce that distribution shift rather than being sampled from
the same distribution as the training data.

Certain algorithms can be used both for interpolation and for extrapolation, depending on how they are applied to the
data.

[`TargetPropertySplit`](@ref) and group-aware strategies like [`GroupShuffleSplit`](@ref) let you deliberately construct
such sets.

## Choosing a strategy

The appropriate splitting strategy depends primarily on the generalisation question you want to answer and on any
structure that must be preserved in the data. For broader discussions of data splitting and performance estimation, see
[crucittiTrain26, arlotSurvey10, cawleyOverfitting10, camachoSet26, vabalasMachine19, josephOptimal22, raschkaModel18](@citet).

| Goal or data structure                                             | Recommended strategy                                                        |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Estimate performance on new i.i.d. observations                    | [`RandomSplit`](@ref), [`KFold`](@ref), or [`ShuffleSplit`](@ref)           |
| Obtain a more stable estimate from a small dataset                 | [`KFold`](@ref), [`RepeatedKFold`](@ref), or [`ShuffleSplit`](@ref)         |
| Preserve class proportions during evaluation                       | [`StratifiedKFold`](@ref)                                                   |
| Evaluate generalisation to unseen groups                           | [`GroupShuffleSplit`](@ref) or [`GroupKFold`](@ref)                         |
| Evaluate prediction of future observations                         | [`TimeSplit`](@ref) or [`TimeSeriesSplit`](@ref)                            |
| Account for temporal dependence or overlapping prediction horizons | [`BlockedCV`](@ref) or [`PurgedKFold`](@ref)                                |
| Select hyperparameters while separately estimating performance     | [`NestedCV`](@ref)                                                          |
| Construct a training set that spans feature space                  | [`KennardStoneSplit`](@ref), [`SPXYSplit`](@ref), or [`OptiSimSplit`](@ref) |
| Construct a space-filling subset for a large dataset               | [`LazyKennardStoneSplit`](@ref) or [`LazyOptiSimSplit`](@ref)               |
| Deliberately test response-range extrapolation                     | [`TargetPropertySplit`](@ref)                                               |

## What DataSplits provides

DataSplits implements these strategies through a common [`partition`](@ref) interface:

- Random and resampling strategies for ordinary train/test splitting and repeated performance estimation.
- Distance-based strategies for selecting training sets that cover the feature or feature–response space.
- Cross-validation strategies, including stratified, nested, group-aware, and time-aware variants.
- Group-aware strategies for keeping dependent observations together.
- Time-series strategies for respecting temporal order and dependence.
- Extrapolation strategies for constructing deliberately out-of-distribution evaluation sets.

Continue to [Getting Started](02-getting-started.md) for the first hands-on example.
