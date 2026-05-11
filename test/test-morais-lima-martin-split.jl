using Random
using Distances
using DataSplits
using Test

@testset "Morais-Lima-Martin Split" begin
  X = rand(3, 50)

  rng = MersenneTwister(123)
  result1 = partition(
    X,
    MoraisLimaMartinSplit(; swap_frac = 0.1, metric = Euclidean());
    train = 80,
    test = 20,
    rng = rng,
  )
  train, test = result1.train, result1.test
  @test is_full_partition(result1, numobs(X)) && cohorts_are_complements(result1, numobs(X))

  # Determinism
  rng1 = MersenneTwister(42)
  r1 = partition(
    X,
    MoraisLimaMartinSplit(; swap_frac = 0.2);
    train = 50,
    test = 50,
    rng = rng1,
  )
  rng2 = MersenneTwister(42)
  r2 = partition(
    X,
    MoraisLimaMartinSplit(; swap_frac = 0.2);
    train = 50,
    test = 50,
    rng = rng2,
  )
  @test r1.train == r2.train
  @test r1.test == r2.test

  @test_throws SplitParameterError MoraisLimaMartinSplit(; swap_frac = 1.5)
end
