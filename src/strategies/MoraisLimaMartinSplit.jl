using Random, Distances

"""
    MoraisLimaMartinSplit(frac; swap_frac=0.1, metric=Euclidean())

Splitting strategy executing Kennard–Stone then randomly swapping a fraction of samples between train and test sets.

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)
- `swap_frac::ValidFraction{T}`: Fraction of samples to swap between train and test (0 < swap_frac < 1)
- `metric::Distances.SemiMetric`: Distance metric for Kennard–Stone (default: Euclidean())
"""
struct MoraisLimaMartinSplit{T,M<:Distances.SemiMetric} <: SplitStrategy
  frac::ValidFraction{T}
  swap_frac::ValidFraction{T}
  metric::M
end

MoraisLimaMartinSplit(frac::Real; swap_frac::Real = 0.1, metric = Euclidean()) =
  MoraisLimaMartinSplit(ValidFraction(frac), ValidFraction(swap_frac), metric)

function _split(data, s::MoraisLimaMartinSplit; rng = Random.GLOBAL_RNG)
  ks = KennardStoneSplit(s.frac, s.metric)
  tt = _split(data, ks; rng = rng)
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
