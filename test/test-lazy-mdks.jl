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

  @test_throws SplitInputError partition(X, LazyMDKSSplit(); train = 70, test = 30)

  result = partition(X, LazyMDKSSplit(); target = y, train = 70, test = 30)
  @test total_size(result) == length(y)
  @test is_disjoint(result)

  result2 = partition(
    X,
    LazyMDKSSplit(; metric = Mahalanobis(cov(X; dims = 2)));
    target = y,
    train = 70,
    test = 30,
  )
  @test total_size(result2) == length(y)
  @test is_disjoint(result2)

  result_mdks = partition(
    X,
    MDKSSplit(; metric = Mahalanobis(cov(X; dims = 2)));
    target = y,
    train = 70,
    test = 30,
  )
  @test Set(result2.train) == Set(result_mdks.train)
  @test Set(result2.test) == Set(result_mdks.test)
  @test is_disjoint(result_mdks)
  @test total_size(result_mdks) == length(y)
end
