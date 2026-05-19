using Test
using Random
using DataSplits
import DataSplits: SplitParameterError

@testset "RepeatedKFold" begin
  N = 30

  @testset "Produces k * n_repeats folds" begin
    cvs = partition(rand(2, N), RepeatedKFold(5; n_repeats = 4))
    @test length(folds(cvs)) == 20
  end

  @testset "Each repeat is a full partition of the data" begin
    cvs = partition(rand(2, N), RepeatedKFold(5; n_repeats = 3); rng = MersenneTwister(0))
    fs = folds(cvs)
    for r = 0:2
      slice = fs[(r*5+1):(r*5+5)]
      test_concat = sort(reduce(vcat, [f.test for f in slice]))
      @test test_concat == 1:N
    end
  end

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

  @testset "Train and test disjoint within each fold" begin
    cvs = partition(rand(2, N), RepeatedKFold(3; n_repeats = 4); rng = MersenneTwister(1))
    for f in folds(cvs)
      @test isempty(intersect(f.train, f.test))
      @test sort(vcat(f.train, f.test)) == 1:N
    end
  end

  @testset "Parameter validation" begin
    @test_throws SplitParameterError partition(rand(2, N), RepeatedKFold(1; n_repeats = 3))
    @test_throws SplitParameterError partition(rand(2, N), RepeatedKFold(5; n_repeats = 0))
  end
end
