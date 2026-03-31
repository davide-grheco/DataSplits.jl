using Random

"""
    RandomSplit{T} <: AbstractSplitStrategy

Randomly splits data into train/test sets.

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)

# Examples
```julia
res = partition(X, RandomSplit(0.8))
X_train, X_test = splitdata(res, X)
```
"""
struct RandomSplit{T} <: AbstractSplitStrategy
  frac::ValidFraction{T}
end

RandomSplit(frac::Real) = RandomSplit(ValidFraction(frac))

consumes(::RandomSplit) = ()
fallback_from_data(::RandomSplit) = ()

function _partition(data, s::RandomSplit; rng, kwargs...)
  N = numobs(data)
  perm = randperm(rng, N)
  cut = floor(Int, s.frac * N)
  return TrainTestSplit(perm[1:cut], perm[cut+1:end])
end
