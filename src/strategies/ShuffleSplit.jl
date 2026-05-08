"""
    ShuffleSplit(n_splits::Integer) <: AbstractCVStrategy

Random permutation cross-validation. For each of the `n_splits`
iterations, observations are randomly shuffled and the requested
`train` and `test` cohort sizes are drawn from the head and the next
slice of the permutation. Mirrors scikit-learn's `ShuffleSplit`, with
the train/test sizes supplied at the `partition` call (in line with
the rest of DataSplits' API).

# Fields
- `n_splits::Int`: Number of resamples (must be ≥ 1).

# Notes
- `train` and `test` sum to `N`, like the rest of the partition API —
  every observation is placed in exactly one cohort per resample.
  (sklearn lets you drop observations by setting `train_size + test_size < 1`;
  this package keeps the "all observations accounted for" invariant.)
- Resamples are independent: an observation can land in train in one
  fold and test in another. This is the defining property of
  `ShuffleSplit` versus `KFold`.

# Examples
```julia
# Fractions.
cvs = partition(X, ShuffleSplit(10); train = 0.8, test = 0.2)

# Absolute counts.
cvs = partition(X, ShuffleSplit(10); train = 80, test = 20)

# Reproducible.
cvs = partition(X, ShuffleSplit(10); train = 0.8, test = 0.2,
                rng = MersenneTwister(42))
```
"""
struct ShuffleSplit <: AbstractResamplingCVStrategy
  n_splits::Int
end

ShuffleSplit(n_splits::Integer) = ShuffleSplit(Int(n_splits))

consumes(::ShuffleSplit) = ()
fallback_from_data(::ShuffleSplit) = ()

function _partition(
  data,
  alg::ShuffleSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  alg.n_splits >= 1 || throw(
    SplitParameterError("ShuffleSplit requires n_splits ≥ 1, got n_splits=$(alg.n_splits)."),
  )

  N = numobs(data)

  result = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.n_splits)
  for i = 1:alg.n_splits
    perm = randperm(rng, N)
    result[i] = TrainTestSplit(perm[1:n_train], perm[(n_train+1):(n_train+n_test)])
  end
  return CrossValidationSplit(result)
end
