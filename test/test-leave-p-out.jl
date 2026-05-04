using Test, DataSplits

@testset "LeavePOut basic contract" begin
  X = randn(2, 6)
  cvs = partition(X, LeavePOut(2))

  @test cvs isa CrossValidationSplit
  @test length(cvs) == binomial(6, 2)  # 15

  for fold in cvs
    @test fold isa TrainTestSplit
    @test length(fold.test) == 2
    @test length(fold.train) == 4
    @test isempty(intersect(fold.train, fold.test))
    @test sort(vcat(fold.train, fold.test)) == collect(1:6)
  end
end

@testset "LeavePOut covers all combinations" begin
  cvs = partition(randn(2, 5), LeavePOut(2))
  test_sets = Set(Set(fold.test) for fold in cvs)
  @test length(test_sets) == binomial(5, 2)  # all unique
end

@testset "LeavePOut splitview and splitdata" begin
  X = randn(2, 5)
  cvs = partition(X, LeavePOut(1))
  for (X_train, X_test) in splitview(cvs, X)
    @test size(X_train, 2) + size(X_test, 2) == 5
  end
  for (X_train, X_test) in splitdata(cvs, X)
    @test size(X_train, 2) + size(X_test, 2) == 5
  end
end

@testset "LeavePOut parameter validation" begin
  X = randn(2, 5)
  @test_throws DataSplits.SplitParameterError partition(X, LeavePOut(0))
  @test_throws DataSplits.SplitParameterError partition(X, LeavePOut(5))
  @test_throws DataSplits.SplitParameterError partition(X, LeavePOut(6))
end

@testset "LeavePOut rejects train/test kwargs" begin
  @test_throws MethodError partition(randn(2, 5), LeavePOut(1); train = 80, test = 20)
end

@testset "LeaveOneOut basic contract" begin
  X = randn(2, 8)
  cvs = partition(X, LeaveOneOut())

  @test cvs isa CrossValidationSplit
  @test length(cvs) == 8

  for (i, fold) in enumerate(cvs)
    @test fold isa TrainTestSplit
    @test length(fold.test) == 1
    @test length(fold.train) == 7
    @test isempty(intersect(fold.train, fold.test))
    @test sort(vcat(fold.train, fold.test)) == collect(1:8)
  end
end

@testset "LeaveOneOut each observation appears as test exactly once" begin
  N = 10
  cvs = partition(randn(2, N), LeaveOneOut())
  test_indices = [fold.test[1] for fold in cvs]
  @test sort(test_indices) == collect(1:N)
end

@testset "LeaveOneOut rejects train/test kwargs" begin
  @test_throws MethodError partition(randn(2, 5), LeaveOneOut(); train = 80, test = 20)
end
