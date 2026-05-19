using Test
using Random
using DataSplits
import DataSplits: SplitParameterError

@testset "RepeatedKFold" begin
  N = 30

  @testset "Repeats differ from each other" begin
    cvs = partition(rand(2, N), RepeatedKFold(5; n_repeats = 2); rng = MersenneTwister(0))
    fs = folds(cvs)
    @test fs[1].test != fs[6].test
  end

  @testset "Reproducible with seeded rng" begin
    a = partition(rand(2, N), RepeatedKFold(5; n_repeats = 3); rng = MersenneTwister(7))
    b = partition(rand(2, N), RepeatedKFold(5; n_repeats = 3); rng = MersenneTwister(7))
    for (x, y) in zip(folds(a), folds(b))
      @test x.train == y.train
      @test x.test == y.test
    end
  end

  @testset "Parameter validation" begin
    @test_throws SplitParameterError partition(rand(2, N), RepeatedKFold(1; n_repeats = 3))
    @test_throws SplitParameterError partition(rand(2, N), RepeatedKFold(5; n_repeats = 0))
  end
end
