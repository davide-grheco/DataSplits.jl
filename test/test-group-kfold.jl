using Test, Random, DataSplits

Random.seed!(42)
X = vcat(randn(30, 2), randn(40, 2) .+ 5, randn(50, 2) .+ 10)'
N = 120
groups = vcat(fill(:a, 30), fill(:b, 40), fill(:c, 50))

@testset "GroupKFold basic contract" begin
  cvs = partition(X, GroupKFold(3); groups = groups)

  @test cvs isa CrossValidationSplit
  @test length(cvs) == 3
  @test length(folds(cvs)) == 3

  for fold in cvs
    @test fold isa TrainTestSplit
    @test isempty(intersect(fold.train, fold.test))
    @test sort(vcat(fold.train, fold.test)) == collect(1:N)
  end
end

@testset "GroupKFold respects group membership" begin
  cvs = partition(X, GroupKFold(3); groups = groups)
  for fold in cvs
    train_groups = unique(groups[fold.train])
    test_groups = unique(groups[fold.test])
    @test isempty(intersect(train_groups, test_groups))
  end
end

@testset "GroupKFold partitions groups across folds" begin
  cvs = partition(X, GroupKFold(3); groups = groups)
  test_groups_per_fold = [Set(unique(groups[fold.test])) for fold in cvs]
  @test reduce(union, test_groups_per_fold) == Set([:a, :b, :c])
  @test sum(length, test_groups_per_fold) == 3
end

@testset "GroupKFold with many groups produces near-balanced folds" begin
  many_groups = repeat(1:30, inner = 4)
  data = randn(2, length(many_groups))
  cvs = partition(data, GroupKFold(5); groups = many_groups)
  fold_sizes = [length(fold.test) for fold in cvs]
  @test sum(fold_sizes) == length(many_groups)
  @test maximum(fold_sizes) - minimum(fold_sizes) <= 4
end

@testset "GroupKFold fallback: ids as both data and groups" begin
  ids = vcat(fill(:a, 40), fill(:b, 40), fill(:c, 40))
  cvs = partition(ids, GroupKFold(3))
  @test length(cvs) == 3
  for fold in cvs
    @test isempty(intersect(fold.train, fold.test))
  end
end

@testset "GroupKFold splitview iterates train/test pairs" begin
  cvs = partition(X, GroupKFold(3); groups = groups)
  pairs = splitview(cvs, X)
  @test length(pairs) == 3
  for (X_train, X_test) in pairs
    @test size(X_train, 2) + size(X_test, 2) == N
  end
end

@testset "GroupKFold splitdata returns concrete subsets" begin
  cvs = partition(X, GroupKFold(3); groups = groups)
  parts = splitdata(cvs, X)
  @test length(parts) == 3
  for (X_train, X_test) in parts
    @test size(X_train, 2) + size(X_test, 2) == N
  end
end

@testset "GroupKFold parameter validation" begin
  @test_throws DataSplits.SplitParameterError partition(X, GroupKFold(1); groups = groups)
  @test_throws DataSplits.SplitParameterError partition(X, GroupKFold(0); groups = groups)
  # k > n_groups
  @test_throws DataSplits.SplitParameterError partition(X, GroupKFold(10); groups = groups)
end

@testset "GroupKFold rejects mismatched groups length" begin
  short_groups = groups[1:50]
  @test_throws DataSplits.SplitInputError partition(X, GroupKFold(3); groups = short_groups)
end

@testset "GroupKFold rejects train/test kwargs" begin
  @test_throws MethodError partition(
    X,
    GroupKFold(3);
    groups = groups,
    train = 80,
    test = 20,
  )
end
