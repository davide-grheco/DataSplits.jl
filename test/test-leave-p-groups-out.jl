using Test, Random, DataSplits

Random.seed!(42)
X = vcat(randn(30, 2), randn(40, 2) .+ 5, randn(50, 2) .+ 10)'
N = 120
groups = vcat(fill(:a, 30), fill(:b, 40), fill(:c, 50))

@testset "LeaveOneGroupOut basic contract" begin
  cvs = partition(X, LeaveOneGroupOut(); groups = groups)

  @test cvs isa CrossValidationSplit
  @test length(cvs) == 3

  for fold in cvs
    @test fold isa TrainTestSplit
    @test isempty(intersect(fold.train, fold.test))
    @test sort(vcat(fold.train, fold.test)) == collect(1:N)
  end
end

@testset "LeaveOneGroupOut isolates one group per fold" begin
  cvs = partition(X, LeaveOneGroupOut(); groups = groups)
  for (f, g) in enumerate(unique(groups))
    fold = folds(cvs)[f]
    @test Set(unique(groups[fold.test])) == Set([g])
    @test g ∉ unique(groups[fold.train])
  end
end

@testset "LeaveOneGroupOut fold count equals number of groups" begin
  many = repeat(1:7, inner = 4)
  data = randn(2, length(many))
  cvs = partition(data, LeaveOneGroupOut(); groups = many)
  @test length(cvs) == 7
end

@testset "LeaveOneGroupOut fallback: ids as both data and groups" begin
  ids = vcat(fill(:a, 10), fill(:b, 10), fill(:c, 10), fill(:d, 10))
  cvs = partition(ids, LeaveOneGroupOut())
  @test length(cvs) == 4
end

@testset "LeaveOneGroupOut requires ≥ 2 groups" begin
  single = fill(:only, 30)
  data = randn(2, 30)
  @test_throws DataSplits.SplitParameterError partition(
    data,
    LeaveOneGroupOut();
    groups = single,
  )
end

@testset "LeaveOneGroupOut rejects mismatched groups length" begin
  short = groups[1:50]
  @test_throws DataSplits.SplitInputError partition(X, LeaveOneGroupOut(); groups = short)
end

@testset "LeaveOneGroupOut fold ordering follows first-occurrence of groups" begin
  ordered = [:c, :c, :a, :a, :b, :b, :a, :c]
  data = randn(2, length(ordered))
  cvs = partition(data, LeaveOneGroupOut(); groups = ordered)
  @test only(unique(ordered[cvs[1].test])) == :c
  @test only(unique(ordered[cvs[2].test])) == :a
  @test only(unique(ordered[cvs[3].test])) == :b
end

@testset "LeaveOneGroupOut() === LeavePGroupsOut(1)" begin
  alg1 = LeaveOneGroupOut()
  alg2 = LeavePGroupsOut(1)
  @test alg1 == alg2
  @test alg1 isa LeavePGroupsOut
end

@testset "LeavePGroupsOut(2) basic contract" begin
  ids = vcat(fill(:a, 5), fill(:b, 5), fill(:c, 5), fill(:d, 5))
  data = randn(2, 20)
  cvs = partition(data, LeavePGroupsOut(2); groups = ids)
  @test cvs isa CrossValidationSplit
  # binomial(4, 2) == 6 folds.
  @test length(cvs) == 6
  for fold in cvs
    @test isempty(intersect(fold.train, fold.test))
    @test sort(vcat(fold.train, fold.test)) == collect(1:20)
    @test length(unique(ids[fold.test])) == 2
  end
end

@testset "LeavePGroupsOut(2) covers every pair exactly once" begin
  ids = vcat(fill(:a, 3), fill(:b, 3), fill(:c, 3), fill(:d, 3))
  data = randn(2, 12)
  cvs = partition(data, LeavePGroupsOut(2); groups = ids)
  test_pairs = [Set(unique(ids[fold.test])) for fold in cvs]
  expected_pairs = Set([
    Set([:a, :b]),
    Set([:a, :c]),
    Set([:a, :d]),
    Set([:b, :c]),
    Set([:b, :d]),
    Set([:c, :d]),
  ])
  @test Set(test_pairs) == expected_pairs
end

@testset "LeavePGroupsOut(2) test cohorts are disjoint from train within fold" begin
  ids = vcat(fill(:a, 5), fill(:b, 5), fill(:c, 5), fill(:d, 5), fill(:e, 5))
  data = randn(2, 25)
  cvs = partition(data, LeavePGroupsOut(2); groups = ids)
  for fold in cvs
    train_groups = Set(unique(ids[fold.train]))
    test_groups = Set(unique(ids[fold.test]))
    @test isempty(intersect(train_groups, test_groups))
  end
end

@testset "LeavePGroupsOut parameter validation" begin
  ids = vcat(fill(:a, 5), fill(:b, 5), fill(:c, 5))
  data = randn(2, 15)
  @test_throws DataSplits.SplitParameterError partition(
    data,
    LeavePGroupsOut(0);
    groups = ids,
  )
  @test_throws DataSplits.SplitParameterError partition(
    data,
    LeavePGroupsOut(3);
    groups = ids,
  )
  @test_throws DataSplits.SplitParameterError partition(
    data,
    LeavePGroupsOut(5);
    groups = ids,
  )
end

@testset "LeavePGroupsOut fallback: ids as both data and groups" begin
  ids = vcat(fill(:a, 4), fill(:b, 4), fill(:c, 4), fill(:d, 4))
  cvs = partition(ids, LeavePGroupsOut(2))
  @test length(cvs) == 6
end
