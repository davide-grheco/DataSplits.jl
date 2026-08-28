```@meta
CurrentModule = DataSplits
```

# Getting Started

Throughout this software and documentation, we use three names for the main roles that data can play in model
development. Data splitting is not intrinsically limited to two or three sets, but a three-way distinction is common.
Terminology varies considerably across scientific disciplines, so we use the following convention:

- **Training set** (_train set_, _training data_, and, especially in chemometrics, _calibration set_): the data used to
  fit or calibrate a model, including estimation of model parameters and any data-dependent preprocessing.
- **Validation set** (_validation data_, _development set_, _dev set_, _tuning set_): the data used during model
  selection, for example to compare candidate models or choose hyperparameters. Cross-validation can be used instead of
  a separate validation set.
- **Test set** (_test data_, _holdout set_, _evaluation set_, _independent test set_): data that play no role in model
  fitting or model selection and are used only to estimate the generalisation performance of the selected modelling
  procedure.

These names are not universal. In some fields, particularly biomedical and applied sciences, _validation set_ or
_external validation set_ may refer to what we call a test set. Conversely, _test set_ is sometimes used for data used
during model development. In this documentation, however, _training_, _validation_, and _test_ always refer to the three
roles defined above.

## Basic API

Most interactions with DataSplits use [`partition`](@ref). It takes the data, a splitting strategy, and any information
required to construct the requested cohorts.

We use the Iris dataset for the first examples:

```@example basic
using DataSplits
using RDatasets
using Random

iris = dataset("datasets", "iris")
X = iris[:, 1:4]
y = iris.Species

size(X), length(y)
```

A simple train/test split is:

```@example basic
res = partition(
    X,
    RandomSplit();
    train = 0.8,
    test = 0.2,
    rng = Xoshiro(42),
)

length(trainindices(res)), length(testindices(res))
```

`partition` returns indices describing the split rather than copies of the data. The corresponding subsets can be
obtained with [`splitdata`](@ref):

```@example basic
X_train, X_test = splitdata(res, X)

size(X_train), size(X_test)
```

DataSplits supports three main forms of `partition`:

```julia
# Train / test
res = partition(data, alg; train, test, kwargs...)

# Train / validation / test
res = partition(
    data,
    test_alg,
    val_alg;
    train,
    validation,
    test,
    kwargs...,
)

# Cross-validation
cvs = partition(data, cv_alg; kwargs...)
```

For a three-way split, `test_alg` first separates the test cohort. `val_alg` then divides the remaining data into
training and validation cohorts. This makes it possible to use different strategies for final evaluation and model
selection.

## Using splits with machine-learning packages

DataSplits returns indices or views rather than imposing a modelling framework. This makes the same split usable with
packages such as MLUtils, Flux, and MLJ.

For arrays, DataSplits follows the Julia ML convention used by MLUtils: observations are stored along the last
dimension. A supervised dataset can therefore be split and passed directly to an `MLUtils.DataLoader`:

```julia
using DataSplits
using MLUtils

res = partition(
    X,
    RandomSplit();
    train = 0.8,
    test = 0.2,
    rng = Xoshiro(42),
)

train_data = trainview(res, X, y)
test_data = testview(res, X, y)

train_loader = DataLoader(
    train_data;
    batchsize = 32,
    shuffle = true,
)

test_loader = DataLoader(
    test_data;
    batchsize = 32,
    shuffle = false,
)
```

When multiple data sources are supplied, trainview and testview return tuples of corresponding views. These can be
consumed directly by DataLoader, keeping predictors and targets aligned.

The resulting loader can be used directly in a Flux training loop:

```julia
using Flux

opt_state = Flux.setup(Adam(), model)

Flux.train!(
    loss,
    model,
    train_loader,
    opt_state,
)
```

Cross-validation results can be converted directly to the (train, test) row-index pairs expected by MLJ:

```julia
using MLJ

cvs = partition(
    X,
    StratifiedKFold(5);
    target = y,
)

mach = machine(model, X, y)

evaluate!(
    mach;
    resampling = rowpairs(cvs),
    measure = accuracy,
)
```

## Cohort sizes

`train`, `validation`, and `test` are specified when calling [`partition`](@ref), rather than when constructing the
splitting strategy. They can be given in three forms:

- **Fractions:** floating-point values in `(0, 1)` that sum to `1.0`.
- **Percentages:** integers that sum to `100`.
- **Counts:** integers that sum to `numobs(data)`.

For example:

```julia
# Fractions
partition(X, RandomSplit(); train = 0.8, test = 0.2)

# Percentages
partition(X, RandomSplit(); train = 80, test = 20)

# Absolute counts
partition(X, RandomSplit(); train = 120, test = 30)  # requires numobs(X) == 150
```

The same convention applies to three-way splits:

```@example basic
threeway = partition(
    X,
    RandomSplit(),
    RandomSplit();
    train = 0.6,
    validation = 0.2,
    test = 0.2,
    rng = Xoshiro(42),
)

(
    train = length(trainindices(threeway)),
    validation = length(valindices(threeway)),
    test = length(testindices(threeway)),
)
```

Cross-validation strategies generally do not take `train` and `test` keywords because fold sizes are determined by the
strategy itself:

```julia
cvs = partition(X, KFold(5))
```

Resampling strategies such as [`ShuffleSplit`](@ref) are an exception: they construct repeated train/test splits and
therefore accept cohort sizes for each repetition.

## Working with a split

`partition` returns split objects containing indices. The corresponding observations can be materialised with
[`splitdata`](@ref) or accessed lazily with [`splitview`](@ref).

### Materialised subsets

[`splitdata`](@ref) returns all cohorts defined by a split:

```@example basic
X_train, X_test = splitdata(res, X)
y_train, y_test = splitdata(res, y)

(size(X_train), length(y_train)), (size(X_test), length(y_test))
```

For a three-way split, it returns training, validation, and test subsets:

Aligned data sources can be split using the same result:

```@example basic
y_train, y_val, y_test = splitdata(threeway, y)

(
    train = length(y_train),
    validation = length(y_val),
    test = length(y_test),
)

```

Individual cohorts can instead be retrieved from several aligned data sources at once with [`traindata`](@ref),
[`valdata`](@ref) and [`testdata`](@ref):

```@example basic
X_train, y_train = traindata(threeway, X, y)
X_val, y_val = valdata(threeway, X, y)
X_test, y_test = testdata(threeway, X, y)

(
    train = (size(X_train), length(y_train)),
    validation = (size(X_val), length(y_val)),
    test = (size(X_test), length(y_test)),
)
```

### Lazy views

When copying the underlying data is unnecessary or expensive, use [`splitview`](@ref):

```@example basic
X_train_view, X_test_view = splitview(res, X)

size(X_train_view), size(X_test_view)
```

The corresponding single-cohort helpers are [`trainview`](@ref), [`valview`](@ref), and [`testview`](@ref):

```@example basic
X_train_view, y_train_view = trainview(res, X, y)

size(X_train_view), length(y_train_view)
```

## Cross-validation

Cross-validation strategies return a [`CrossValidationSplit`](@ref) containing one train/test split for each fold.

```@example basic
cvs = partition(X, KFold(5))

length(folds(cvs))
```

The folds can be used directly with [`splitview`](@ref):

```@example basic
for (X_tr, X_te) in splitview(cvs, X)
    println(length(X_tr), " train / ", length(X_te), " test")
end
```

The same split can be applied to aligned feature and target data:

```@example basic
for fold in folds(cvs)
    X_tr, y_tr = trainview(fold, X, y)
    X_te, y_te = testview(fold, X, y)

    println(length(X_tr), " train / ", length(X_te), " test")
end
```

[`trainview`](@ref) and [`testview`](@ref) also accept the whole [`CrossValidationSplit`](@ref) rather than a single
fold. They then return one tuple of co-indexed views per fold, which is convenient when the folds are needed up front,
for instance to hand each one to a data loader:

```@example basic
train_folds = trainview(cvs, X, y)
X_fold1, y_fold1 = train_folds[1]

(folds = length(train_folds), first_fold = (length(X_fold1), length(y_fold1)))
```

The materialising counterparts [`traindata`](@ref) and [`testdata`](@ref) behave the same way but copy the observations
instead of viewing them.

When performing model selection with cross-validation, any data-dependent preprocessing must also be fitted within the
training portion of each fold.

## Additional information

Some splitting strategies require information in addition to the data being split. These inputs are passed as keyword
arguments to [`partition`](@ref):

- `target` — response, property, or class labels, used by strategies such as [`SPXYSplit`](@ref) and
  [`StratifiedKFold`](@ref).
- `time` — timestamps or another temporal ordering variable, used by strategies such as [`TimeSplit`](@ref),
  [`TimeSeriesSplit`](@ref), and [`BlockedCV`](@ref).
- `groups` — group membership, used by strategies such as [`GroupShuffleSplit`](@ref) and [`GroupKFold`](@ref).

For example:

```julia
partition(X, StratifiedKFold(5); target = classes)
partition(X, GroupKFold(5); groups = patient_ids)
partition(X, TimeSeriesSplit(5); time = timestamps)
```

### Single-input shorthand

If the data being split are themselves the values required by the strategy, the corresponding keyword can be omitted.

For example, when splitting a vector of timestamps:

```julia
partition(
    timestamps,
    TimeSplit();
    train = 0.7,
    test = 0.3,
)
```

Likewise, group identifiers can be split directly:

```julia
partition(
    patient_ids,
    GroupShuffleSplit();
    train = 0.8,
    test = 0.2,
)
```

## Data formats

`data` can be a matrix, a vector, a Tables.jl-compatible container such as a `DataFrame`, or any custom type
implementing the `MLUtils` observation interface.

For matrices, DataSplits follows the Julia machine-learning convention:

> **columns are samples and rows are features**

Tables such as `DataFrame`s instead store observations in rows. For example:

```@example basic
X_rows = Matrix(iris[:, 1:4])
X_matrix = permutedims(X_rows)

(size(X_rows), size(X_matrix))
```

The first matrix has 150 rows × 4 features, whereas the matrix passed to matrix-oriented DataSplits strategies has 4
features × 150 observations.

If your data store samples in rows, transpose or permute them before passing them to strategies that operate on feature
matrices:

For Tables.jl-compatible inputs, rows are interpreted as observations. DataSplits performs the required conversion
internally when a strategy needs a feature-by-observation matrix.

All returned indices refer to observations and are positive integers in `1:N`.

For custom containers or arrays with non-standard indexing, prefer [`splitdata`](@ref) and [`splitview`](@ref) over
indexing with the returned indices manually.

## Reproducibility

Strategies that involve randomisation, such as [`RandomSplit`](@ref), [`ShuffleSplit`](@ref), and
[`GroupShuffleSplit`](@ref), accept an `rng` keyword.

Using the same seeded random-number generator state reproduces the same split:

```@example basic
split_a = partition(
    X,
    RandomSplit();
    train = 0.8,
    test = 0.2,
    rng = Xoshiro(42),
)

split_b = partition(
    X,
    RandomSplit();
    train = 0.8,
    test = 0.2,
    rng = Xoshiro(42),
)

split_c = partition(
    X,
    RandomSplit();
    train = 0.8,
    test = 0.2,
    rng = Xoshiro(43),
)

(trainindices(split_a) == trainindices(split_b),
trainindices(split_a) == trainindices(split_c))
```

Using an explicit random-number generator makes the source of randomness visible in scripts, examples, and reproducible
analyses.

## Accessing indices directly

When you need the indices themselves, use the public accessor functions rather than reading fields of the returned split
object directly:

```@example basic
(
    train = length(trainindices(threeway)),
    validation = length(valindices(threeway)),
    test = length(testindices(threeway)),
)
```

Cross-validation results can be inspected with [`folds`](@ref).

These accessors form part of the public API and should be preferred over relying on the internal representation of split
objects.

## Where to go next

The examples above cover the common DataSplits workflow:

1. Choose a splitting strategy,
2. Call [`partition`](@ref),
3. Obtain data subsets or views,
4. Fit, select, and evaluate models using the appropriate cohorts.

For guidance on choosing an appropriate strategy, see [Why Splitting Matters](@ref).

For the complete behaviour of [`partition`](@ref), result types, and helper functions, continue to the
[Core API Reference](@ref).
