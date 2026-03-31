using Test
using Distances
using Statistics
using DataSplits
import DataSplits: SplitInputError

@testset "LazyMDKS splitter" begin
  X = [
    4 1 9 5 5 7
    10 9 3 3 8 2
    8 7 2 7 2 1
    6 8 2 2 6 10
    2 1 4 3 6 10
    2 10 6 4 1 9
  ]'
  y = [4, 1, 7, 5, 2, 5]

  @test_throws SplitInputError partition(X, LazyMDKSSplit(0.7))

  result = partition(X, LazyMDKSSplit(0.7); target = y)
  train_idx, test_idx = result.train, result.test
  @test length(train_idx) + length(test_idx) == length(y)
  @test isempty(intersect(train_idx, test_idx))

  result2 =
    partition(X, LazyMDKSSplit(0.7; metric = Mahalanobis(cov(X; dims = 2))); target = y)
  train_idx2, test_idx2 = result2.train, result2.test
  @test length(train_idx2) + length(test_idx2) == length(y)
  @test isempty(intersect(train_idx2, test_idx2))

  result_mdks =
    partition(X, MDKSSplit(0.7; metric = Mahalanobis(cov(X; dims = 2))); target = y)
  @test Set(train_idx2) == Set(result_mdks.train)
  @test Set(test_idx2) == Set(result_mdks.test)
  @test isempty(intersect(result_mdks.train, result_mdks.test))
  @test length(result_mdks.train) + length(result_mdks.test) == length(y)
end
