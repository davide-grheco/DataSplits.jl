using Random
using Distances
using DataSplits
using Test

@testset "Morais-Lima-Martin Split" begin
  X = rand(3, 50)

  rng = MersenneTwister(123)
  result1 =
    split(X, MoraisLimaMartinSplit(0.8; swap_frac = 0.1, metric = Euclidean()); rng = rng)
  train, test = result1.train, result1.test
  @test sort(vcat(train, test)) == 1:numobs(X)
  @test isempty(intersect(train, test))

  # Determinism
  rng1 = MersenneTwister(42)
  r1 = split(X, MoraisLimaMartinSplit(0.5; swap_frac = 0.2); rng = rng1)
  rng2 = MersenneTwister(42)
  r2 = split(X, MoraisLimaMartinSplit(0.5; swap_frac = 0.2); rng = rng2)
  @test r1.train == r2.train
  @test r1.test == r2.test

  @test_throws SplitParameterError MoraisLimaMartinSplit(0.5; swap_frac = 1.0)
end
