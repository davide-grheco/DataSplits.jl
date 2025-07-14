using Random
using StatsBase

export RandomSplit

struct RandomSplit{T} <: SplitStrategy
  frac::ValidFraction{T}
end
RandomSplit(frac::Real) = RandomSplit(ValidFraction(frac))

function _split(data, s::RandomSplit; rng)
  idx_range = axes(data, 1)
  first_idx = first(idx_range)
  N = length(idx_range)

  perm = randperm(rng, N)
  cut = floor(Int, s.frac * N)

  train = (first_idx - 1) .+ perm[1:cut]
  test = (first_idx - 1) .+ perm[cut+1:end]

  return train, test
end
