using Test, Random, DataSplits
using Statistics: mean

Random.seed!(42)

@testset "StratifiedKFold classification: every fold sees every class" begin
  N = 90
  labels = vcat(fill(:a, 30), fill(:b, 30), fill(:c, 30))
  X = randn(2, N)
  cvs = partition(X, StratifiedKFold(5); target = labels)

  @test cvs isa CrossValidationSplit
  @test length(cvs) == 5

  for fold in cvs
    @test isempty(intersect(fold.train, fold.test))
    @test sort(vcat(fold.train, fold.test)) == collect(1:N)
    test_labels = labels[fold.test]
    @test Set(unique(test_labels)) == Set([:a, :b, :c])
  end
end

@testset "StratifiedKFold classification: balanced folds" begin
  labels = vcat(fill(:a, 50), fill(:b, 50))
  X = randn(2, 100)
  cvs = partition(X, StratifiedKFold(5); target = labels)
  for fold in cvs
    counts = [count(==(c), labels[fold.test]) for c in (:a, :b)]
    @test all(==(10), counts)
  end
end

@testset "StratifiedKFold imbalanced classification" begin
  labels = vcat(fill(:rare, 10), fill(:common, 90))
  X = randn(2, 100)
  cvs = partition(X, StratifiedKFold(5); target = labels)
  for fold in cvs
    @test count(==(:rare), labels[fold.test]) == 2
    @test count(==(:common), labels[fold.test]) == 18
  end
end

@testset "StratifiedKFold integer labels treated as classes" begin
  labels = repeat([0, 1, 2], inner = 25)
  X = randn(2, 75)
  cvs = partition(X, StratifiedKFold(5); target = labels)
  for fold in cvs
    @test Set(unique(labels[fold.test])) == Set([0, 1, 2])
  end
end

@testset "StratifiedKFold regression: quantile binning" begin
  Random.seed!(7)
  y = randn(200)
  X = randn(2, 200)
  cvs = partition(X, StratifiedKFold(5; bins = 4); target = y)

  for fold in cvs
    @test isempty(intersect(fold.train, fold.test))
    train_mean, test_mean = mean(y[fold.train]), mean(y[fold.test])
    @test abs(train_mean - test_mean) < 0.5
  end
end

@testset "StratifiedKFold fallback: target as both data and target" begin
  labels = repeat(['a', 'b', 'c'], inner = 20)
  cvs = partition(labels, StratifiedKFold(4))
  @test length(cvs) == 4
end

@testset "StratifiedKFold splitview iterates train/test pairs" begin
  labels = vcat(fill(:a, 25), fill(:b, 25))
  X = randn(2, 50)
  cvs = partition(X, StratifiedKFold(5); target = labels)
  for (X_train, X_test) in splitview(cvs, X)
    @test size(X_train, 2) + size(X_test, 2) == 50
  end
end

@testset "StratifiedKFold parameter validation" begin
  labels = vcat(fill(:a, 10), fill(:b, 10))
  X = randn(2, 20)
  @test_throws DataSplits.SplitParameterError partition(
    X,
    StratifiedKFold(1);
    target = labels,
  )
  @test_throws DataSplits.SplitParameterError partition(
    X,
    StratifiedKFold(5; bins = 1);
    target = randn(20),
  )
end

@testset "StratifiedKFold rejects classes with too few members" begin
  labels = vcat(fill(:a, 30), fill(:b, 3))
  X = randn(2, 33)
  @test_throws DataSplits.SplitParameterError partition(
    X,
    StratifiedKFold(5);
    target = labels,
  )
end

@testset "StratifiedKFold rejects mismatched target length" begin
  labels = vcat(fill(:a, 30), fill(:b, 30))
  X = randn(2, 60)
  short = labels[1:30]
  @test_throws DataSplits.SplitInputError partition(X, StratifiedKFold(3); target = short)
end

@testset "StratifiedKFold accepts BitVector / Bool targets" begin
  labels = BitVector(vcat(falses(30), trues(30)))
  X = randn(2, 60)
  cvs = partition(X, StratifiedKFold(5); target = labels)
  for fold in cvs
    @test count(labels[fold.test]) == 6
    @test count(.!labels[fold.test]) == 6
  end
end

@testset "StratifiedKFold tolerates dense duplicate-value continuous targets" begin
  Random.seed!(11)
  y = vcat(zeros(80), randn(20))
  X = randn(2, 100)
  cvs = partition(X, StratifiedKFold(5; bins = 4); target = y)
  for fold in cvs
    @test isempty(intersect(fold.train, fold.test))
    @test sort(vcat(fold.train, fold.test)) == collect(1:100)
  end
end
