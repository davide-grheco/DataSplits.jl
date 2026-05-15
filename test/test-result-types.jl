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

@testset "trainview / testview — TrainTestSplit single-arg" begin
  X = rand(3, 10)
  r = TrainTestSplit([1, 2, 3, 4, 5, 6, 7, 8], [9, 10])

  tv = trainview(r, X)
  @test tv == X[:, r.train]
  @test parent(tv) === X          # obsview — shares memory

  tev = testview(r, X)
  @test tev == X[:, r.test]
  @test parent(tev) === X
end

@testset "trainview / testview — TrainTestSplit multi-arg" begin
  X = rand(3, 10)
  y = rand(10)
  r = TrainTestSplit([1, 2, 3, 4, 5, 6, 7, 8], [9, 10])

  Xtr, ytr = trainview(r, X, y)
  @test Xtr == X[:, r.train]
  @test ytr == y[r.train]
  @test parent(Xtr) === X
  @test parent(ytr) === y

  Xte, yte = testview(r, X, y)
  @test Xte == X[:, r.test]
  @test yte == y[r.test]
end

@testset "trainview / valview / testview — TrainValTestSplit" begin
  X = rand(3, 10)
  y = rand(10)
  r = TrainValTestSplit([1, 2, 3, 4, 5, 6, 7], [8], [9, 10])

  @test trainview(r, X) == X[:, r.train]
  @test valview(r, X) == X[:, r.val]
  @test testview(r, X) == X[:, r.test]

  Xtr, ytr = trainview(r, X, y)
  Xval, yval = valview(r, X, y)
  Xte, yte = testview(r, X, y)
  @test Xtr == X[:, r.train]
  @test ytr == y[r.train]
  @test Xval == X[:, r.val]
  @test yval == y[r.val]
  @test Xte == X[:, r.test]
  @test yte == y[r.test]
end

@testset "traindata / testdata — TrainTestSplit" begin
  X = rand(3, 10)
  y = rand(10)
  r = TrainTestSplit([1, 2, 3, 4, 5, 6, 7, 8], [9, 10])

  Xtr = traindata(r, X)
  @test Xtr == X[:, r.train]
  @test !(Xtr === X)              # getobs — materialised copy

  Xtr2, ytr2 = traindata(r, X, y)
  @test Xtr2 == X[:, r.train]
  @test ytr2 == y[r.train]

  Xte2, yte2 = testdata(r, X, y)
  @test Xte2 == X[:, r.test]
  @test yte2 == y[r.test]
end

@testset "traindata / valdata / testdata — TrainValTestSplit" begin
  X = rand(3, 10)
  y = rand(10)
  r = TrainValTestSplit([1, 2, 3, 4, 5, 6, 7], [8], [9, 10])

  @test traindata(r, X) == X[:, r.train]
  @test valdata(r, X) == X[:, r.val]
  @test testdata(r, X) == X[:, r.test]

  Xtr, ytr = traindata(r, X, y)
  Xval, yval = valdata(r, X, y)
  Xte, yte = testdata(r, X, y)
  @test Xtr == X[:, r.train]
  @test ytr == y[r.train]
  @test Xval == X[:, r.val]
  @test yval == y[r.val]
  @test Xte == X[:, r.test]
  @test yte == y[r.test]
end

@testset "trainview / testview — CrossValidationSplit" begin
  X = rand(3, 10)
  y = rand(10)
  folds_vec = [
    TrainTestSplit([1, 2, 3, 4, 5, 6, 7], [8, 9, 10]),
    TrainTestSplit([1, 2, 3, 8, 9, 10], [4, 5, 6, 7]),
  ]
  cvs = CrossValidationSplit(folds_vec)

  tv = trainview(cvs, X)
  @test length(tv) == 2
  for (i, v) in enumerate(tv)
    @test v == X[:, folds_vec[i].train]
  end

  tev = testview(cvs, X)
  @test length(tev) == 2
  for (i, v) in enumerate(tev)
    @test v == X[:, folds_vec[i].test]
  end

  # multi-source
  tv2 = trainview(cvs, X, y)
  @test length(tv2) == 2
  for (i, (Xv, yv)) in enumerate(tv2)
    @test Xv == X[:, folds_vec[i].train]
    @test yv == y[folds_vec[i].train]
  end
end

@testset "traindata / testdata — CrossValidationSplit" begin
  X = rand(3, 10)
  y = rand(10)
  folds_vec = [
    TrainTestSplit([1, 2, 3, 4, 5, 6, 7], [8, 9, 10]),
    TrainTestSplit([1, 2, 3, 8, 9, 10], [4, 5, 6, 7]),
  ]
  cvs = CrossValidationSplit(folds_vec)

  td = traindata(cvs, X)
  @test length(td) == 2
  for (i, v) in enumerate(td)
    @test v == X[:, folds_vec[i].train]
  end

  td2 = traindata(cvs, X, y)
  @test length(td2) == 2
  for (i, (Xv, yv)) in enumerate(td2)
    @test Xv == X[:, folds_vec[i].train]
    @test yv == y[folds_vec[i].train]
  end
end
