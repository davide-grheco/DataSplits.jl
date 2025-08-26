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
  result_max = DataSplits.split(
    X,
    MaximumDissimilaritySplit(0.6; distance_cutoff = 0.0, metric = Euclidean()),
    rng = Random.seed!(42),
  )
  train_max, test_max = result_max.train, result_max.test
  @test length(train_max) == 3
  @test isempty(intersect(train_max, test_max))
  @test length(train_max) + length(test_max) == 5

  # Minimum Dissimilarity
  result_min = DataSplits.split(
    X,
    MinimumDissimilaritySplit(0.6; distance_cutoff = 0.0, metric = Euclidean()),
    rng = Random.seed!(42),
  )
  train_min, test_min = result_min.train, result_min.test
  @test length(train_min) == 3
  @test isempty(intersect(train_min, test_min))
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
  result_max = DataSplits.split(
    X,
    LazyMaximumDissimilaritySplit(0.6; distance_cutoff = 0.0, metric = Euclidean()),
    rng = Random.seed!(42),
  )
  train_max, test_max = result_max.train, result_max.test
  @test length(train_max) == 3
  @test isempty(intersect(train_max, test_max))
  @test length(train_max) + length(test_max) == 5

  # Minimum Dissimilarity
  result_min = DataSplits.split(
    X,
    LazyMinimumDissimilaritySplit(0.6; distance_cutoff = 0.0, metric = Euclidean()),
    rng = Random.seed!(42),
  )
  train_min, test_min = result_min.train, result_min.test
  @test length(train_min) == 3
  @test isempty(intersect(train_min, test_min))
  @test length(train_min) + length(test_min) == 5
end
