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

@testset "CrossValidationSplit splitview" begin
  X = rand(3, 10)
  folds_vec = [
    TrainTestSplit([1, 2, 3, 4, 5, 6, 7], [8, 9, 10]),
    TrainTestSplit([1, 2, 3, 8, 9, 10], [4, 5, 6, 7]),
  ]
  cvs = CrossValidationSplit(folds_vec)
  sv = splitview(cvs, X)
  @test length(sv) == 2
  for (i, (X_train, X_test)) in enumerate(sv)
    @test X_train == X[:, folds_vec[i].train]
    @test X_test == X[:, folds_vec[i].test]
  end
end
@testset "rowpairs" begin
  folds_vec = [
    TrainTestSplit([1, 2, 3], [4, 5]),
    TrainTestSplit([4, 5, 1], [2, 3]),
    TrainTestSplit([2, 3, 4], [1, 5]),
  ]
  cvs = CrossValidationSplit(folds_vec)
  pairs = rowpairs(cvs)

  @test pairs isa Vector
  @test length(pairs) == 3
  @test all(p -> p isa Tuple{Vector{Int},Vector{Int}}, pairs)
  for (i, (tr, te)) in enumerate(pairs)
    @test tr == folds_vec[i].train
    @test te == folds_vec[i].test
  end
end
