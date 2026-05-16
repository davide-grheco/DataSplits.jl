using Test
using Random
using DataSplits
import DataSplits: SplitParameterError

@testset "GroupShuffleSplitCV" begin
  N = 120
  X = randn(2, N)
  groups = vcat(fill(:a, 40), fill(:b, 40), fill(:c, 40))

  @testset "Produces n_splits folds" begin
    cvs = partition(
      X,
      GroupShuffleSplitCV(10);
      groups = groups,
      train = 80,
      test = 40,
      rng = MersenneTwister(0),
    )
    @test length(folds(cvs)) == 10
  end

  @testset "Each fold is a full partition with no group leakage" begin
    cvs = partition(
      X,
      GroupShuffleSplitCV(8);
      groups = groups,
      train = 80,
      test = 40,
      rng = MersenneTwister(1),
    )
    for f in folds(cvs)
      @test is_disjoint(f)
      @test total_size(f) == N
      @test no_group_leakage(f, groups)
    end
  end

  @testset "Resamples are independent (different from GroupKFold)" begin
    cvs = partition(
      X,
      GroupShuffleSplitCV(30);
      groups = groups,
      train = 80,
      test = 40,
      rng = MersenneTwister(2),
    )
    # Across resamples a group should appear in test more than once,
    # not exactly once like in GroupKFold.
    test_counts = Dict(g => 0 for g in unique(groups))
    for f in folds(cvs)
      for g in unique(groups[testindices(f)])
        test_counts[g] += 1
      end
    end
    @test maximum(values(test_counts)) > 1
  end

  @testset "Train size may overshoot (whole-group invariant)" begin
    cvs = partition(
      X,
      GroupShuffleSplitCV(5);
      groups = groups,
      train = 50,
      test = 70,
      rng = MersenneTwister(3),
    )
    for f in folds(cvs)
      # groups are size 40 each, so train fills with a multiple of 40 ≥ 50.
      @test length(trainindices(f)) >= 50
      @test total_size(f) == N
    end
  end

  @testset "Reproducible with seeded rng" begin
    a = partition(
      X,
      GroupShuffleSplitCV(5);
      groups = groups,
      train = 80,
      test = 40,
      rng = MersenneTwister(7),
    )
    b = partition(
      X,
      GroupShuffleSplitCV(5);
      groups = groups,
      train = 80,
      test = 40,
      rng = MersenneTwister(7),
    )
    for (x, y) in zip(folds(a), folds(b))
      @test same_indices(x, y)
    end
  end

  @testset "Fallback: ids as both data and groups" begin
    ids = vcat(fill(:a, 40), fill(:b, 40), fill(:c, 40))
    cvs = partition(
      ids,
      GroupShuffleSplitCV(4);
      train = 80,
      test = 40,
      rng = MersenneTwister(11),
    )
    @test length(folds(cvs)) == 4
    for f in folds(cvs)
      @test total_size(f) == N
      @test is_disjoint(f)
    end
  end

  @testset "n_splits validation" begin
    @test_throws SplitParameterError partition(
      X,
      GroupShuffleSplitCV(0);
      groups = groups,
      train = 80,
      test = 40,
    )
  end

  @testset "Missing train/test raises (UndefKeywordError)" begin
    @test_throws UndefKeywordError partition(
      X,
      GroupShuffleSplitCV(5);
      groups = groups,
    )
  end
end
