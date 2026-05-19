using Test
using Random
using DataSplits
import DataSplits: SplitParameterError

@testset "ShuffleSplit" begin
  X = rand(4, 50)

  @testset "Resamples are independent (different from KFold)" begin
    cvs = partition(X, ShuffleSplit(20); train = 0.8, test = 0.2, rng = MersenneTwister(0))
    test_counts = zeros(Int, 50)
    for f in folds(cvs)
      for i in f.test
        test_counts[i] += 1
      end
    end
    @test sum(test_counts) == 20 * 10
    @test maximum(test_counts) > 1
  end

  @testset "Missing train/test raises (UndefKeywordError)" begin
    @test_throws UndefKeywordError partition(X, ShuffleSplit(5))
    @test_throws UndefKeywordError partition(X, ShuffleSplit(5); train = 0.8)
    @test_throws UndefKeywordError partition(X, ShuffleSplit(5); test = 0.2)
  end

  @testset "n_splits validation" begin
    @test_throws SplitParameterError partition(X, ShuffleSplit(0); train = 0.8, test = 0.2)
  end

  @testset "splitview round-trip" begin
    cvs = partition(X, ShuffleSplit(3); train = 0.8, test = 0.2)
    for (Xtr, Xte) in splitview(cvs, X)
      @test size(Xtr, 1) == 4
      @test size(Xte, 1) == 4
      @test size(Xtr, 2) + size(Xte, 2) == 50
    end
  end
end
