using Test
using Distances
using Statistics
using DataSplits
import DataSplits: SplitInputError, SplitParameterError, SplitNotImplementedError

@testset "LazySPXY splitter" begin
  X = [
    4 1 9 5 5 7
    10 9 3 3 8 2
    8 7 2 7 2 1
    6 8 2 2 6 10
    2 1 4 3 6 10
    2 10 6 4 1 9
  ]'
  y = [4, 1, 7, 5, 2, 5]

  @test_throws SplitParameterError split(X, LazySPXYSplit(0.7))

  result = split((X, y), LazySPXYSplit(0.7))
  train_idx, test_idx = result.train, result.test

  result2 =
    split((X, y), LazySPXYSplit(0.7; metric_X = Cityblock(), metric_y = Euclidean()))
  train_idx2, test_idx2 = result2.train, result2.test
  @test length(train_idx2) + length(test_idx2) == length(y)
  @test isempty(intersect(train_idx2, test_idx2))

  # Compare with standard SPXY for small data
  result_spxy = split((X, y), SPXYSplit(0.7))
  @test Set(train_idx) == Set(result_spxy.train)
  @test Set(test_idx) == Set(result_spxy.test)
  @test isempty(intersect(train_idx, test_idx))
  @test length(train_idx) + length(test_idx) == length(y)
end
