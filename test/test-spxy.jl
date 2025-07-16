using Test
using Distances
using DataSplits


rowset(M) = Set(eachrow(M))

@testset "SPXY splitter" begin
  # Tests obtained from astartes
  # https://github.com/JacksonBurns/astartes/blob/32ef58293c3205771062284e550b1da8509af8dd/test/unit/samplers/interpolative/test_spxy.py
  X = [
    4 1 9 5 5 7
    10 9 3 3 8 2
    8 7 2 7 2 1
    6 8 2 2 6 10
    2 1 4 3 6 10
    2 10 6 4 1 9
  ]
  y = [4, 1, 7, 5, 2, 5]
  labels = ["one", "two", "three", "four", "five", "six"]


  @test_throws MethodError split(X, SPXYSplit(0.7))

  train_idx, test_idx = split((X, y), SPXYSplit(0.7; metric = Euclidean()))

  expected_train = Set([2, 3, 5, 6])
  expected_test = Set([1, 4])

  @test Set(train_idx) == expected_train
  @test Set(test_idx) == expected_test
  @test isempty(intersect(train_idx, test_idx))
  @test length(train_idx) + length(test_idx) == length(y)
end
