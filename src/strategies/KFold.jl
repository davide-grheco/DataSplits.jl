"""
    KFold(k::Integer; shuffle::Bool=false) <: AbstractCVStrategy

Standard k-fold cross-validation. Splits the dataset into `k` roughly
equal folds; each fold takes a turn as the test set while the remaining
folds form the training set. Equivalent in spirit to scikit-learn's `KFold`.

# Fields
- `k::Int`: Number of folds (must be ≥ 2 and ≤ number of observations).
- `shuffle::Bool`: When `true`, observations are randomly permuted before
  folding using the `rng` passed to `partition`, so different seeds yield
  different fold assignments. When `false` (default), observations are
  assigned in order and the split is fully deterministic.

# Notes
- Fold sizes differ by at most 1 observation (first `n mod k` folds are
  one sample larger), mirroring scikit-learn's behaviour.
- Unlike `GroupKFold`, no extra keyword arguments are required.

# Examples
```julia
# Deterministic (default).
cvs = partition(X, KFold(5))

# Shuffled — different seeds give different fold assignments.
cvs = partition(X, KFold(5; shuffle = true); rng = MersenneTwister(42))

for (X_train, X_test) in splitview(cvs, X)
    # train and evaluate
end
```
"""
struct KFold <: AbstractCVStrategy
  k::Int
  shuffle::Bool
end

KFold(k::Integer; shuffle::Bool = false) = KFold(Int(k), shuffle)

consumes(::KFold) = ()
fallback_from_data(::KFold) = ()

function _partition(data, alg::KFold; rng = Random.default_rng(), kwargs...)
  alg.k >= 2 || throw(SplitParameterError("KFold requires k ≥ 2, got k=$(alg.k)."))

  N = numobs(data)
  alg.k <= N || throw(
    SplitParameterError("KFold(k=$(alg.k)) requires at least k observations; got $N."),
  )

  indices = alg.shuffle ? randperm(rng, N) : collect(1:N)

  fold_size, remainder = divrem(N, alg.k)

  fold_test = Vector{Vector{Int}}(undef, alg.k)
  offset = 0
  for f = 1:alg.k
    len = fold_size + (f <= remainder ? 1 : 0)
    fold_test[f] = indices[(offset+1):(offset+len)]
    offset += len
  end

  result = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.k)
  for f = 1:alg.k
    test_set = Set(fold_test[f])
    train_idx = [i for i in indices if !(i in test_set)]
    result[f] = TrainTestSplit(train_idx, fold_test[f])
  end
  return CrossValidationSplit(result)
end
