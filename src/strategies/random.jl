using Random

"""
    RandomSplit <: AbstractSplitStrategy

Randomly splits data into the requested cohort sizes.

# Examples
```julia
res = partition(X, RandomSplit(); train = 80, test = 20)
X_train, X_test = splitdata(res, X)
```
"""
struct RandomSplit <: AbstractSplitStrategy end

consumes(::RandomSplit) = ()
fallback_from_data(::RandomSplit) = ()

function _partition(data, ::RandomSplit; n_train, n_test, rng, kwargs...)
  N = numobs(data)
  perm = randperm(rng, N)
  return TrainTestSplit(perm[1:n_train], perm[(n_train+1):end])
end
