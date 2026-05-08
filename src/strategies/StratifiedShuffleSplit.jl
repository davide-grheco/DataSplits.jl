"""
    StratifiedShuffleSplit(n_splits::Integer; bins::Integer=10) <: AbstractResamplingCVStrategy

Stratified resampling cross-validation. Combines the per-resample
random-draw structure of [`ShuffleSplit`](@ref) with the
class/quantile-bin balancing of [`StratifiedKFold`](@ref): each
resample preserves the global class proportions in both train and
test cohorts.

Targets are passed via the `target=` keyword (or, by fallback, `data`
itself plays that role).

# Fields
- `n_splits::Int`: Number of resamples (must be ≥ 1).
- `bins::Int`: Number of quantile bins used when `target` is
  floating-point (must be ≥ 2; default `10`). Ignored for non-float
  targets, which are treated as discrete classes.

# Stratification rule
Same as `StratifiedKFold`:
- **Discrete `target`** (e.g. `Int`, `Bool`, `Symbol`, `String`): each
  unique value defines a class.
- **Float `target`**: targets are binned into `bins` quantile-based bins.

Within each class/bin, members are randomly shuffled and the global
train fraction is applied locally — `round(Int, n_train * |class| / N)`
go to train, the rest to test. Rounding remainder is absorbed in the
last class processed so totals match `n_train` / `n_test` exactly.

# Notes
- `n_train + n_test == N` (project-wide invariant — sklearn allows
  dropping observations, this package does not).
- Each class needs at least 2 members so that both cohorts can
  receive a representative; otherwise a `SplitParameterError` is
  raised. Reduce `bins` for continuous targets if you hit it.

# Examples
```julia
# Classification.
cvs = partition(X, StratifiedShuffleSplit(10);
                target = labels, train = 0.8, test = 0.2)

# Regression: 10 quantile bins (default).
cvs = partition(X, StratifiedShuffleSplit(10);
                target = y_continuous, train = 0.8, test = 0.2)
```
"""
struct StratifiedShuffleSplit <: AbstractResamplingCVStrategy
  n_splits::Int
  bins::Int
end

StratifiedShuffleSplit(n_splits::Integer; bins::Integer = 10) =
  StratifiedShuffleSplit(Int(n_splits), Int(bins))

consumes(::StratifiedShuffleSplit) = (:target,)
fallback_from_data(::StratifiedShuffleSplit) = (:target,)

function _partition(
  data,
  alg::StratifiedShuffleSplit;
  n_train,
  n_test,
  target,
  rng = Random.default_rng(),
  kwargs...,
)
  alg.n_splits >= 1 || throw(
    SplitParameterError(
      "StratifiedShuffleSplit requires n_splits ≥ 1, got n_splits=$(alg.n_splits).",
    ),
  )
  alg.bins >= 2 || throw(
    SplitParameterError(
      "StratifiedShuffleSplit requires bins ≥ 2, got bins=$(alg.bins).",
    ),
  )

  N = numobs(data)
  classes = _stratification_classes(target, alg.bins)

  unique_classes = unique(classes)
  members_by_class = Dict(c => findall(==(c), classes) for c in unique_classes)
  for (c, m) in members_by_class
    length(m) >= 2 || throw(
      SplitParameterError(
        "StratifiedShuffleSplit: class/bin $(repr(c)) has only $(length(m)) member(s); reduce `bins` or use a different target.",
      ),
    )
  end

  result = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.n_splits)
  for s = 1:alg.n_splits
    train_idx = Int[]
    test_idx = Int[]
    # Track how many slots have already been allocated so the last class
    # can absorb rounding remainder and totals match n_train / n_test exactly.
    allocated_train = 0
    allocated_test = 0
    classes_ordered = collect(unique_classes)
    for (j, c) in enumerate(classes_ordered)
      members = copy(members_by_class[c])
      shuffle!(rng, members)
      if j == length(classes_ordered)
        n_train_c = n_train - allocated_train
        n_test_c = n_test - allocated_test
      else
        n_train_c = round(Int, n_train * length(members) / N)
        # Guarantee both cohorts get at least one member from each class.
        n_train_c = clamp(n_train_c, 1, length(members) - 1)
        n_test_c = length(members) - n_train_c
      end
      append!(train_idx, members[1:n_train_c])
      append!(test_idx, members[(n_train_c+1):(n_train_c+n_test_c)])
      allocated_train += n_train_c
      allocated_test += n_test_c
    end
    result[s] = TrainTestSplit(train_idx, test_idx)
  end
  return CrossValidationSplit(result)
end
