using Test
using DataSplits
using Random
using Distances

@testset "Minimum/Maximum Dissimilarity Split" begin
  X = [
    0 0 0 0 0
    1 0 0 0 0
    1 1 0 0 0
    1 1 1 0 0
    1 1 1 1 0
  ]'

  # Maximum Dissimilarity
  result_max = DataSplits.partition(
    X,
    MaximumDissimilaritySplit(; distance_cutoff = 0.0, metric = Euclidean());
    train = 60,
    test = 40,
    rng = Random.seed!(42),
  )
  train_max, test_max = result_max.train, result_max.test
  @test length(train_max) == 3
  @test is_disjoint(result_max)
  @test length(train_max) + length(test_max) == 5

  # Minimum Dissimilarity
  result_min = DataSplits.partition(
    X,
    MinimumDissimilaritySplit(; distance_cutoff = 0.0, metric = Euclidean());
    train = 60,
    test = 40,
    rng = Random.seed!(42),
  )
  train_min, test_min = result_min.train, result_min.test
  @test length(train_min) == 3
  @test is_disjoint(result_min)
  @test length(train_min) + length(test_min) == 5
end


@testset "Lazy Minimum/Maximum Dissimilarity Split" begin
  X = [
    0 0 0 0 0
    1 0 0 0 0
    1 1 0 0 0
    1 1 1 0 0
    1 1 1 1 0
  ]'

  # Maximum Dissimilarity
  result_max = DataSplits.partition(
    X,
    LazyMaximumDissimilaritySplit(; distance_cutoff = 0.0, metric = Euclidean());
    train = 60,
    test = 40,
    rng = Random.seed!(42),
  )
  train_max, test_max = result_max.train, result_max.test
  @test length(train_max) == 3
  @test is_disjoint(result_max)
  @test length(train_max) + length(test_max) == 5

  # Minimum Dissimilarity
  result_min = DataSplits.partition(
    X,
    LazyMinimumDissimilaritySplit(; distance_cutoff = 0.0, metric = Euclidean());
    train = 60,
    test = 40,
    rng = Random.seed!(42),
  )
  train_min, test_min = result_min.train, result_min.test
  @test length(train_min) == 3
  @test is_disjoint(result_min)
  @test length(train_min) + length(test_min) == 5
end
@testset "Dissimilarity constructor validation" begin
  import DataSplits: SplitParameterError
  @test_throws SplitParameterError MaximumDissimilaritySplit(; distance_cutoff = -0.1)
  @test_throws SplitParameterError LazyMaximumDissimilaritySplit(; distance_cutoff = -0.1)
  @test_throws SplitParameterError LazyMinimumDissimilaritySplit(; distance_cutoff = -0.1)
  # MinimumDissimilaritySplit delegates to OptiSimSplit, so same check applies
  @test_throws SplitParameterError MinimumDissimilaritySplit(; distance_cutoff = -0.1)
end
