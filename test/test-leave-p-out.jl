using Test, DataSplits


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

@testset "LeaveOneOut rejects train/test kwargs" begin
  @test_throws MethodError partition(randn(2, 5), LeaveOneOut(); train = 80, test = 20)
end
