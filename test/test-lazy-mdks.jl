using Test
using Distances
using Statistics
using DataSplits

@testset "LazyMDKS splitter" begin
  # Small synthetic dataset
  X = [
    4 1 9 5 5 7
    10 9 3 3 8 2
    8 7 2 7 2 1
    6 8 2 2 6 10
    2 1 4 3 6 10
    2 10 6 4 1 9
  ]'
  y = [4, 1, 7, 5, 2, 5]

  # Should throw if called with only X
  @test_throws MethodError split(X, LazyMDKSSplit(0.7))

  # Default: Mahalanobis for X, Euclidean for y
  result = split((X, y), LazyMDKSSplit(0.7))
  train_idx, test_idx = result.train, result.test
  @test length(train_idx) + length(test_idx) == length(y)
  @test isempty(intersect(train_idx, test_idx))

  # Custom metric for X
  result2 = split((X, y), LazyMDKSSplit(0.7; metric = Mahalanobis(cov(X; dims = 2))))
  train_idx2, test_idx2 = result2.train, result2.test
  @test length(train_idx2) + length(test_idx2) == length(y)
  @test isempty(intersect(train_idx2, test_idx2))

  # Compare with standard MDKS for small data
  result_mdks = split((X, y), MDKSSplit(0.7; metric = Mahalanobis(cov(X; dims = 2))))
  @test Set(train_idx2) == Set(result_mdks.train)
  @test Set(test_idx2) == Set(result_mdks.test)
  @test isempty(intersect(result_mdks.train, result_mdks.test))
  @test length(result_mdks.train) + length(result_mdks.test) == length(y)
end
