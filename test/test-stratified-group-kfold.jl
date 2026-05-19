using Test
using Random
using DataSplits
import DataSplits: SplitParameterError, SplitInputError

@testset "StratifiedGroupKFold" begin
  @testset "Groups respected (no group spans folds)" begin
    # 30 obs, 10 groups of 3, binary labels.
    groups = repeat(1:10, inner = 3)
    labels = vcat(fill(:a, 15), fill(:b, 15))
    X = randn(2, 30)

    cvs = partition(X, StratifiedGroupKFold(5); target = labels, groups = groups)

    @test length(folds(cvs)) == 5
    for f in folds(cvs)
      train_groups = unique(groups[f.train])
      test_groups = unique(groups[f.test])
      @test isempty(intersect(train_groups, test_groups))
      @test sort(vcat(f.train, f.test)) == 1:30
    end
  end

  @testset "Each obs tests exactly once across folds" begin
    groups = repeat(1:10, inner = 3)
    labels = vcat(fill(1, 15), fill(2, 15))
    cvs = partition(rand(2, 30), StratifiedGroupKFold(5); target = labels, groups = groups)
    test_concat = sort(reduce(vcat, [f.test for f in folds(cvs)]))
    @test test_concat == 1:30
  end

  @testset "Class balance across folds (within tolerance)" begin
    rng = MersenneTwister(0)
    # 60 obs, 20 groups of 3, ratio 40/20 of class :a/:b.
    groups = repeat(1:20, inner = 3)
    labels = shuffle(rng, vcat(fill(:a, 40), fill(:b, 20)))
    cvs = partition(rand(2, 60), StratifiedGroupKFold(4); target = labels, groups = groups)
    # Each test fold should hold roughly 10 :a and 5 :b.
    for f in folds(cvs)
      n_a = count(==(:a), labels[f.test])
      n_b = count(==(:b), labels[f.test])
      @test 7 <= n_a <= 13
      @test 2 <= n_b <= 8
    end
  end

  @testset "Continuous target with quantile bins" begin
    rng = MersenneTwister(1)
    groups = repeat(1:30, inner = 2)
    y = randn(rng, 60)
    cvs =
      partition(rand(2, 60), StratifiedGroupKFold(5; bins = 4); target = y, groups = groups)
    @test length(folds(cvs)) == 5
    for f in folds(cvs)
      train_groups = unique(groups[f.train])
      test_groups = unique(groups[f.test])
      @test isempty(intersect(train_groups, test_groups))
    end
  end

  @testset "Rare class covered under heavy imbalance" begin
    # 200 obs, 50 groups of 4. Class :b is 5% of total — under count-based
    # scoring it gets drowned by :a and may not appear in every fold.
    # With proportion-normalised scoring (counts / class_total) it should.
    rng = MersenneTwister(7)
    groups = repeat(1:50, inner = 4)
    labels = shuffle(rng, vcat(fill(:a, 190), fill(:b, 10)))
    cvs = partition(rand(2, 200), StratifiedGroupKFold(5); target = labels, groups = groups)
    for f in folds(cvs)
      @test count(==(:b), labels[f.test]) >= 1
    end
  end

  @testset "Reproducible with shuffle and rng" begin
    groups = repeat(1:10, inner = 3)
    labels = vcat(fill(1, 15), fill(2, 15))
    X = rand(2, 30)
    cvs1 = partition(
      X,
      StratifiedGroupKFold(3; shuffle = true);
      target = labels,
      groups = groups,
      rng = MersenneTwister(99),
    )
    cvs2 = partition(
      X,
      StratifiedGroupKFold(3; shuffle = true);
      target = labels,
      groups = groups,
      rng = MersenneTwister(99),
    )
    for (a, b) in zip(folds(cvs1), folds(cvs2))
      @test a.train == b.train
      @test a.test == b.test
    end
  end

  @testset "Deterministic without shuffle" begin
    groups = repeat(1:10, inner = 3)
    labels = vcat(fill(1, 15), fill(2, 15))
    X = rand(2, 30)
    cvs1 = partition(X, StratifiedGroupKFold(3); target = labels, groups = groups)
    cvs2 = partition(X, StratifiedGroupKFold(3); target = labels, groups = groups)
    for (a, b) in zip(folds(cvs1), folds(cvs2))
      @test a.train == b.train
      @test a.test == b.test
    end
  end

  @testset "Parameter validation" begin
    groups = repeat(1:10, inner = 3)
    labels = vcat(fill(1, 15), fill(2, 15))
    @test_throws SplitParameterError partition(
      rand(2, 30),
      StratifiedGroupKFold(1);
      target = labels,
      groups = groups,
    )
    @test_throws SplitParameterError partition(
      rand(2, 30),
      StratifiedGroupKFold(3; bins = 1);
      target = randn(30),
      groups = groups,
    )
    # k > unique groups
    @test_throws SplitParameterError partition(
      rand(2, 30),
      StratifiedGroupKFold(11);
      target = labels,
      groups = groups,
    )
  end

  @testset "Both target and groups are required (no fallback)" begin
    @test_throws SplitInputError partition(
      rand(2, 30),
      StratifiedGroupKFold(3);
      target = vcat(fill(1, 15), fill(2, 15)),
    )
    @test_throws SplitInputError partition(
      rand(2, 30),
      StratifiedGroupKFold(3);
      groups = repeat(1:10, inner = 3),
    )
  end
end
