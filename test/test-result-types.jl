using Test
using DataSplits
using Random

@testset "show" begin
  X = rand(3, 10)
  res2 = partition(X, RandomSplit(); train = 80, test = 20, rng = MersenneTwister(1))
  res3 = partition(
    X,
    RandomSplit(),
    RandomSplit();
    train = 70,
    validation = 10,
    test = 20,
    rng = MersenneTwister(2),
  )

  @test sprint(show, res2) == "TrainTestSplit  train: 8 obs  test: 2 obs"
  @test sprint(show, res3) == "TrainValTestSplit  train: 7 obs  val: 1 obs  test: 2 obs"
end

@testset "iterate / destructuring" begin
  X = rand(3, 100)
  res2 = partition(X, RandomSplit(); train = 80, test = 20, rng = MersenneTwister(3))
  train, test = res2
  @test train == res2.train
  @test test == res2.test
  @test length(res2) == 2
  @test collect(res2) == [res2.train, res2.test]

  res3 = partition(
    X,
    RandomSplit(),
    RandomSplit();
    train = 70,
    validation = 10,
    test = 20,
    rng = MersenneTwister(4),
  )
  train, val, test = res3
  @test train == res3.train
  @test val == res3.val
  @test test == res3.test
  @test length(res3) == 3
  @test collect(res3) == [res3.train, res3.val, res3.test]
end

@testset "CrossValidationSplit indexing" begin
  folds_vec = [TrainTestSplit([1, 2], [3]), TrainTestSplit([1, 3], [2])]
  cvs = CrossValidationSplit(folds_vec)
  @test cvs[1] === folds_vec[1]
  @test cvs[2] === folds_vec[2]
  @test cvs[end] === folds_vec[2]
  @test first(cvs) === folds_vec[1]
  @test last(cvs) === folds_vec[2]
  sub = cvs[1:1]
  @test sub isa CrossValidationSplit
  @test length(sub) == 1
  @test eltype(typeof(cvs)) <: TrainTestSplit
end
