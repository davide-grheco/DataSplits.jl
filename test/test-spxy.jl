using Test
using Random
using Distances
using Statistics
using MLUtils
using DataSplits
import DataSplits: SplitInputError, SplitParameterError, SplitNotImplementedError

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
  ]'
  y = [4, 1, 7, 5, 2, 5]

  # 70% train, 30% test on N=6 → n_train=4, n_test=2 (sum=100)
  @test_throws SplitInputError partition(X, SPXYSplit(); train = 70, test = 30)
  @test_throws SplitInputError partition(X, LazySPXYSplit(); train = 70, test = 30)

  result = partition(X, SPXYSplit(); target = y, train = 70, test = 30)
  train_idx, test_idx = result.train, result.test

  result2 = partition(
    X,
    SPXYSplit(; metric_X = Cityblock(), metric_y = Euclidean());
    target = y,
    train = 70,
    test = 30,
  )

  expected_train = Set([2, 3, 5, 6])
  expected_test = Set([1, 4])

  @test Set(train_idx) == expected_train
  @test Set(test_idx) == expected_test
end

@testset "SPXY more variables than samples" begin
  X = [
    4 1 9 5 5
    10 9 3 3 8
    8 7 2 7 2
    6 8 2 2 6
    2 1 4 3 6
    2 10 6 4 1
  ]
  y = [4, 1, 7, 5, 2]

  result = partition(X, SPXYSplit(); target = y, train = 70, test = 30)
  train_idx, test_idx = result.train, result.test

  result2 = partition(
    X,
    SPXYSplit(; metric_X = Cityblock(), metric_y = Euclidean());
    target = y,
    train = 70,
    test = 30,
  )

  expected_train = Set([2, 3, 5, 1])
  expected_test = Set([4])

  @test Set(train_idx) == expected_train
  @test Set(test_idx) == expected_test
end

@testset "MDKS splitter" begin
  rng = MersenneTwister(7)
  X = randn(rng, 5, 120)
  X[1, :] .*= 8
  X[2, :] .+= 0.9 .* X[1, :]
  y = randn(rng, 120)

  reference = partition(
    X,
    SPXYSplit(; metric_X = Mahalanobis(inv(cov(X; dims = 2))), metric_y = Euclidean());
    target = y,
    train = 0.8,
    test = 0.2,
  )

  @test_throws SplitInputError partition(X, MDKSSplit(); train = 70, test = 30)

  eager = partition(X, MDKSSplit(); target = y, train = 0.8, test = 0.2)
  lazy = partition(X, LazyMDKSSplit(); target = y, train = 0.8, test = 0.2)

  @test trainindices(eager) == trainindices(reference)
  @test trainindices(lazy) == trainindices(reference)
  @test testindices(eager) == testindices(reference)

  # Passing the covariance rather than its inverse is a different metric, and
  # must not silently agree.
  wrong = partition(
    X,
    SPXYSplit(; metric_X = Mahalanobis(cov(X; dims = 2)), metric_y = Euclidean());
    target = y,
    train = 0.8,
    test = 0.2,
  )
  @test trainindices(wrong) != trainindices(reference)

  # A singular covariance, here from more features than samples, must still
  # produce a split rather than failing to factor.
  Xdeg = randn(MersenneTwister(8), 6, 6)
  ydeg = randn(MersenneTwister(9), 6)
  degenerate = partition(Xdeg, MDKSSplit(); target = ydeg, train = 70, test = 30)
  @test length(trainindices(degenerate)) == 4
  @test length(testindices(degenerate)) == 2
  @test sort(vcat(trainindices(degenerate), testindices(degenerate))) == 1:6
end

@testset "LazySPXYSplit stays allocation-free in its inner loop" begin
  X = randn(Xoshiro(1), 8, 200)
  y = randn(Xoshiro(2), 200)

  @test all(isconcretetype, fieldtypes(typeof(DataSplits.XYObsTable(X, y))))
  @test all(
    isconcretetype,
    fieldtypes(typeof(DataSplits.LazySPXYMetric(Euclidean(), Euclidean(), 1.0, 1.0))),
  )

  split(Xm, ym) = partition(Xm, LazySPXYSplit(); target = ym, train = 0.8, test = 0.2)
  split(X, y)                                     # compile before measuring
  lazy_bytes = @allocated split(X, y)

  eager(Xm, ym) = partition(Xm, SPXYSplit(); target = ym, train = 0.8, test = 0.2)
  eager(X, y)
  eager_bytes = @allocated eager(X, y)

  @test lazy_bytes < eager_bytes

  @test trainindices(split(X, y)) == trainindices(eager(X, y))
end
