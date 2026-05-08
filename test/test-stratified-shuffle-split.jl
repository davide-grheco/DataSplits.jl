using Test
using Random
using DataSplits
import DataSplits: SplitParameterError

@testset "StratifiedShuffleSplit" begin
  @testset "Classification — class proportions preserved per resample" begin
    rng = MersenneTwister(0)
    labels = vcat(fill(:a, 30), fill(:b, 50), fill(:c, 20))
    X = randn(4, 100)
    cvs = partition(X, StratifiedShuffleSplit(20);
                    target = labels, train = 0.8, test = 0.2,
                    rng = rng)

    @test length(folds(cvs)) == 20
    for f in folds(cvs)
      @test length(f.train) == 80
      @test length(f.test) == 20
      @test isempty(intersect(f.train, f.test))
      @test sort(vcat(f.train, f.test)) == 1:100

      # Each class proportionally represented in train and test (within rounding).
      train_labels = labels[f.train]
      test_labels = labels[f.test]
      for c in (:a, :b, :c)
        global_p = count(==(c), labels) / 100
        train_p = count(==(c), train_labels) / 80
        test_p = count(==(c), test_labels) / 20
        @test abs(train_p - global_p) <= 0.05
        @test abs(test_p - global_p) <= 0.10
      end
    end
  end

  @testset "Regression — quantile binning" begin
    rng = MersenneTwister(1)
    y = randn(rng, 100)
    X = randn(rng, 4, 100)
    cvs = partition(X, StratifiedShuffleSplit(5; bins = 4);
                    target = y, train = 0.7, test = 0.3,
                    rng = MersenneTwister(2))
    for f in folds(cvs)
      @test length(f.train) == 70
      @test length(f.test) == 30
      @test isempty(intersect(f.train, f.test))
    end
  end

  @testset "Resamples differ across iterations" begin
    rng = MersenneTwister(42)
    labels = vcat(fill(1, 50), fill(2, 50))
    cvs = partition(rand(2, 100), StratifiedShuffleSplit(5);
                    target = labels, train = 0.8, test = 0.2,
                    rng = rng)
    folds_list = folds(cvs)
    @test folds_list[1].train != folds_list[2].train
  end

  @testset "Reproducible with rng" begin
    labels = vcat(fill(1, 30), fill(2, 70))
    X = randn(2, 100)
    cvs1 = partition(X, StratifiedShuffleSplit(3); target = labels,
                     train = 0.8, test = 0.2, rng = MersenneTwister(7))
    cvs2 = partition(X, StratifiedShuffleSplit(3); target = labels,
                     train = 0.8, test = 0.2, rng = MersenneTwister(7))
    for (a, b) in zip(folds(cvs1), folds(cvs2))
      @test a.train == b.train
      @test a.test == b.test
    end
  end

  @testset "Fallback: target as both data and target" begin
    labels = vcat(fill(:a, 50), fill(:b, 50))
    cvs = partition(labels, StratifiedShuffleSplit(2);
                    train = 0.8, test = 0.2, rng = MersenneTwister(0))
    for f in folds(cvs)
      @test length(f.train) + length(f.test) == 100
    end
  end

  @testset "Class with too few members raises" begin
    labels = vcat(fill(:a, 50), [:rare])  # 1 member of :rare
    @test_throws SplitParameterError partition(
      randn(2, 51),
      StratifiedShuffleSplit(2);
      target = labels,
      train = 0.8,
      test = 0.2,
    )
  end

  @testset "Missing train/test raises UndefKeywordError" begin
    @test_throws UndefKeywordError partition(rand(2, 50), StratifiedShuffleSplit(3);
                                             target = vcat(fill(1, 25), fill(2, 25)))
  end

  @testset "n_splits and bins validation" begin
    labels = vcat(fill(1, 50), fill(2, 50))
    @test_throws SplitParameterError partition(rand(2, 100), StratifiedShuffleSplit(0);
                                                target = labels, train = 0.8, test = 0.2)
    @test_throws SplitParameterError partition(rand(2, 100), StratifiedShuffleSplit(3; bins = 1);
                                                target = randn(100), train = 0.8, test = 0.2)
  end
end
