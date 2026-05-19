using Test, Random, DataSplits
import DataSplits: SplitParameterError, SplitNotImplementedError

@testset "NestedCV" begin
  N = 60
  X = randn(2, N)

  @testset "KFold × KFold basic contract" begin
    cvs = partition(X, NestedCV(KFold(5), KFold(3)))
    @test length(folds(cvs)) == 5
    for outerfold in folds(cvs)
      @test outerfold isa NestedFold
      @test isempty(intersect(trainindices(outerfold), testindices(outerfold)))
      @test sort(vcat(trainindices(outerfold), testindices(outerfold))) == 1:N
      inner = innerfolds(outerfold)
      @test length(folds(inner)) == 3
      for innerfold in folds(inner)
        @test isempty(intersect(innerfold.train, innerfold.test))
        # Inner indices are absolute and a subset of the outer training cohort.
        @test issubset(Set(innerfold.train), Set(trainindices(outerfold)))
        @test issubset(Set(innerfold.test), Set(trainindices(outerfold)))
        # Inner train ∪ test == outer train (KFold partitions the pool).
        @test sort(vcat(innerfold.train, innerfold.test)) ==
              sort(trainindices(outerfold))
      end
    end
  end

  @testset "Outer test cohorts tile 1:N disjointly" begin
    cvs = partition(X, NestedCV(KFold(5), KFold(3)))
    test_concat = sort(reduce(vcat, [testindices(f) for f in folds(cvs)]))
    @test test_concat == 1:N
  end

  @testset "splitdata / splitview on NestedFold" begin
    cvs = partition(X, NestedCV(KFold(5), KFold(3)))
    f = folds(cvs)[1]
    Xtr, Xte = splitdata(f, X)
    @test size(Xtr, 2) + size(Xte, 2) == N
    Xtr_v, Xte_v = splitview(f, X)
    @test size(Xtr_v, 2) == length(trainindices(f))
    @test size(Xte_v, 2) == length(testindices(f))
  end

  @testset "Iteration: train, test = outerfold" begin
    cvs = partition(X, NestedCV(KFold(5), KFold(3)))
    for outerfold in folds(cvs)
      tr, te = outerfold
      @test tr == trainindices(outerfold)
      @test te == testindices(outerfold)
    end
  end

  @testset "Group-aware nesting (slot resolution union)" begin
    groups = vcat(fill(:a, 15), fill(:b, 15), fill(:c, 15), fill(:d, 15))
    cvs = partition(X, NestedCV(GroupKFold(4), GroupKFold(3)); groups = groups)
    for outerfold in folds(cvs)
      # No group leakage outer.
      @test isempty(
        intersect(
          Set(groups[trainindices(outerfold)]),
          Set(groups[testindices(outerfold)]),
        ),
      )
      for innerfold in folds(innerfolds(outerfold))
        @test isempty(
          intersect(Set(groups[innerfold.train]), Set(groups[innerfold.test])),
        )
      end
    end
  end

  @testset "Stratified inner CV uses sliced target" begin
    y = vcat(zeros(Int, 30), ones(Int, 30))
    cvs = partition(X, NestedCV(KFold(5), StratifiedKFold(3)); target = y)
    for outerfold in folds(cvs)
      # Inner folds should be roughly class-balanced within the outer-train pool.
      pool_y = y[trainindices(outerfold)]
      for innerfold in folds(innerfolds(outerfold))
        train_y = y[innerfold.train]
        test_y = y[innerfold.test]
        @test 0 in train_y && 1 in train_y
        @test 0 in test_y && 1 in test_y
      end
    end
  end

  @testset "Inner resampling strategy rejected" begin
    @test_throws SplitParameterError partition(X, NestedCV(KFold(5), ShuffleSplit(3)))
  end

  @testset "Reproducibility with seeded rng" begin
    a = partition(X, NestedCV(KFold(5), KFold(3)); rng = MersenneTwister(42))
    b = partition(X, NestedCV(KFold(5), KFold(3)); rng = MersenneTwister(42))
    for (fa, fb) in zip(folds(a), folds(b))
      @test trainindices(fa) == trainindices(fb)
      @test testindices(fa) == testindices(fb)
      for (ia, ib) in zip(folds(innerfolds(fa)), folds(innerfolds(fb)))
        @test ia.train == ib.train
        @test ia.test == ib.test
      end
    end
  end
end
