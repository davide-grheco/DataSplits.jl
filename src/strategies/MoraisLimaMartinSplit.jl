using Random, Distances

"""
    MoraisLimaMartinSplit(; swap_frac=0.1, metric=Euclidean())

Kennard–Stone initialisation followed by random swapping of a fraction of
samples between train and test sets.

# Fields
- `swap_frac::Float64`: Fraction of samples to swap (0 < swap_frac < 1)
- `metric::Distances.SemiMetric`: Distance metric for Kennard–Stone (default: `Euclidean()`)

# Examples
```julia
res = partition(X, MoraisLimaMartinSplit(); train=80, test=20)
res = partition(X, MoraisLimaMartinSplit(; swap_frac=0.05); train=80, test=20)
```
"""
struct MoraisLimaMartinSplit{M<:Distances.SemiMetric} <: AbstractSplitStrategy
  swap_frac::Float64
  metric::M
end

function MoraisLimaMartinSplit(; swap_frac::Real = 0.1, metric = Euclidean())
  0 < swap_frac < 1 || throw(
    SplitParameterError("`swap_frac` must be strictly between 0 and 1, got $swap_frac."),
  )
  MoraisLimaMartinSplit(Float64(swap_frac), metric)
end

consumes(::MoraisLimaMartinSplit) = (:data,)
fallback_from_data(::MoraisLimaMartinSplit) = ()

function _partition(
  data,
  s::MoraisLimaMartinSplit;
  n_train,
  n_test,
  rng = Random.GLOBAL_RNG,
  kwargs...,
)
  ks = KennardStoneSplit(s.metric)
  tt = _partition(data, ks; n_train = n_train, n_test = n_test, rng = rng)
  train_idx = copy(tt.train)
  test_idx = copy(tt.test)
  n_swap = round(Int, s.swap_frac * min(length(train_idx), length(test_idx)))
  if n_swap > 0
    perm_train = randperm(rng, length(train_idx))
    train_sel = train_idx[perm_train[1:n_swap]]
    perm_test = randperm(rng, length(test_idx))
    test_sel = test_idx[perm_test[1:n_swap]]
    train_idx = setdiff(train_idx, train_sel)
    append!(train_idx, test_sel)
    test_idx = setdiff(test_idx, test_sel)
    append!(test_idx, train_sel)
  end
  return TrainTestSplit(train_idx, test_idx)
end
