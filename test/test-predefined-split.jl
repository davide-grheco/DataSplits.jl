using Test
using DataSplits
import DataSplits: SplitInputError, SplitParameterError

@testset "PredefinedSplit" begin
  X = rand(4, 12)

  @testset "Basic three-fold assignment" begin
    test_fold = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2]
    cvs = partition(X, PredefinedSplit(test_fold))

    @test length(folds(cvs)) == 3
    # Each observation tests exactly once.
    test_concat = reduce(vcat, [f.test for f in folds(cvs)])
    @test sort(test_concat) == 1:12
    # Within a fold, train ∩ test is empty and train ∪ test = 1:12.
    for f in folds(cvs)
      @test isempty(intersect(f.train, f.test))
      @test sort(vcat(f.train, f.test)) == 1:12
    end
  end

  @testset "Negative IDs mean train-only" begin
    # Obs 7-12 are training-only.
    test_fold = [0, 0, 1, 1, 2, 2, -1, -1, -1, -1, -1, -1]
    cvs = partition(X, PredefinedSplit(test_fold))

    @test length(folds(cvs)) == 3
    # Train-only indices appear in every fold's train cohort.
    for f in folds(cvs)
      @test all(i -> i in f.train, 7:12)
      @test isempty(intersect(f.test, 7:12))
    end
    # Test indices across folds cover only obs 1-6.
    test_concat = sort(reduce(vcat, [f.test for f in folds(cvs)]))
    @test test_concat == 1:6
  end

  @testset "Fold IDs need not be contiguous" begin
    test_fold = [10, 10, 20, 20, 50, 50]
    cvs = partition(rand(2, 6), PredefinedSplit(test_fold))
    @test length(folds(cvs)) == 3
    # Folds emitted in ascending order of ID.
    @test folds(cvs)[1].test == [1, 2]
    @test folds(cvs)[2].test == [3, 4]
    @test folds(cvs)[3].test == [5, 6]
  end

  @testset "Length mismatch raises SplitInputError" begin
    @test_throws SplitInputError partition(X, PredefinedSplit([0, 0, 1, 1]))
  end

  @testset "All-negative fold IDs raise SplitParameterError" begin
    @test_throws SplitParameterError partition(
      X,
      PredefinedSplit(fill(-1, 12)),
    )
  end

  @testset "Construction accepts any integer vector" begin
    cvs = partition(X, PredefinedSplit(Int8[0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2]))
    @test length(folds(cvs)) == 3
  end

  @testset "splitview round-trip" begin
    test_fold = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2]
    cvs = partition(X, PredefinedSplit(test_fold))
    for (Xtr, Xte) in splitview(cvs, X)
      @test size(Xtr, 1) == 4
      @test size(Xte, 1) == 4
      @test size(Xtr, 2) + size(Xte, 2) == 12
    end
  end
end
