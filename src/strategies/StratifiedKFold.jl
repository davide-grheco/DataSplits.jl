using Statistics: quantile

"""
    StratifiedKFold(k::Integer; bins::Integer=10) <: AbstractCVStrategy

Stratified k-fold cross-validation. Within each fold, every class (or
quantile bin) is represented in roughly the same proportion as in the
full dataset.

Targets are passed via the `target=` keyword (or, by fallback, `data` itself
plays that role).

# Fields
- `k::Int`: Number of folds (must be ≥ 2).
- `bins::Int`: Number of quantile bins used when `target` is floating-point
  (must be ≥ 2; default `10`). Ignored for non-float targets, which are
  treated as discrete classes.

# Stratification rule
- **Discrete `target`** (e.g. `Int`, `Bool`, `Symbol`, `String`): each unique
  value defines a class.
- **Float `target`**: targets are binned into `bins` quantile-based bins;
  each bin defines a class.

Within each class, indices are distributed round-robin across the `k`
folds, so every fold gets a near-equal share of every class.

# Notes
- Each class needs at least `k` members; otherwise a `SplitParameterError`
  is raised. For discrete targets this is a hard constraint; for binned
  continuous targets, lower `bins` if you hit it.
- When a continuous target has many repeated values (e.g. lots of zeros),
  quantile edges may collapse and effectively yield fewer than `bins`
  populated bins. Stratification still works, but bin coverage is uneven.

# Examples
```julia
# Classification.
cvs = partition(X, StratifiedKFold(5); target = labels)

# Regression: 10 quantile bins (default).
cvs = partition(X, StratifiedKFold(5); target = y_continuous)

# Regression with a custom number of bins.
cvs = partition(X, StratifiedKFold(5; bins = 4); target = y_continuous)
```
"""
struct StratifiedKFold <: AbstractCVStrategy
  k::Int
  bins::Int
end

StratifiedKFold(k::Integer; bins::Integer = 10) = StratifiedKFold(Int(k), Int(bins))

consumes(::StratifiedKFold) = (:target,)
fallback_from_data(::StratifiedKFold) = (:target,)

function _partition(data, alg::StratifiedKFold; target, kwargs...)
  alg.k >= 2 ||
    throw(SplitParameterError("StratifiedKFold requires k ≥ 2, got k=$(alg.k)."))
  alg.bins >= 2 ||
    throw(SplitParameterError("StratifiedKFold requires bins ≥ 2, got bins=$(alg.bins)."))

  N = numobs(data)
  # Anticipates issue #22 (length validation belongs in `partition`).
  length(target) == N || throw(
    SplitInputError(
      "`target` length ($(length(target))) does not match number of observations ($N).",
    ),
  )

  classes = _stratification_classes(target, alg.bins)

  fold_test = [Int[] for _ = 1:alg.k]
  for c in unique(classes)
    members = findall(==(c), classes)
    length(members) >= alg.k || throw(
      SplitParameterError(
        "StratifiedKFold(k=$(alg.k)): class/bin $(repr(c)) has only $(length(members)) members; reduce k or bins.",
      ),
    )
    for (j, idx) in enumerate(members)
      f = mod1(j, alg.k)
      push!(fold_test[f], idx)
    end
  end

  result = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.k)
  for f = 1:alg.k
    test_set = Set(fold_test[f])
    train_idx = [i for i = 1:N if !(i in test_set)]
    result[f] = TrainTestSplit(train_idx, fold_test[f])
  end
  return CrossValidationSplit(result)
end

# Float targets get binned by quantile; everything else is treated as a class.
_stratification_classes(target::AbstractVector{<:AbstractFloat}, bins::Integer) =
  _quantile_bin(target, bins)
_stratification_classes(target::AbstractVector, _bins::Integer) = target

function _quantile_bin(target::AbstractVector{<:AbstractFloat}, bins::Integer)
  edges = quantile(target, range(0, 1; length = bins + 1))
  inner = edges[2:(end-1)]
  [searchsortedfirst(inner, t) for t in target]
end
