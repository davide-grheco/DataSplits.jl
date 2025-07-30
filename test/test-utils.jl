using Test
using DataSplits: train_test_counts, SplitParameterError, ValidFraction

@testset "train_test_counts" begin
  # Normal case
  @test train_test_counts(10, 0.7) == (7, 3)
  @test train_test_counts(10, 0.3) == (3, 7)
  # Edge: minimum allowed
  @test train_test_counts(4, 0.5) == (2, 2)
  # Throws if not enough samples
  @test_throws SplitParameterError train_test_counts(1, 0.5)
  # Throws if fraction out of bounds
  @test_throws SplitParameterError train_test_counts(10, 0.0)
  @test_throws SplitParameterError train_test_counts(10, 1.0)
  @test_throws SplitParameterError train_test_counts(10, -0.1)
  @test_throws SplitParameterError train_test_counts(10, 1.1)
  # Throws if split would result in too few samples
  @test_throws SplitParameterError train_test_counts(10, 0.01)
  @test_throws SplitParameterError train_test_counts(10, 0.95)
  # Custom minimums
  @test train_test_counts(10, 0.6; min_train = 3, min_test = 3) == (6, 4)
  @test_throws SplitParameterError train_test_counts(5, 0.6; min_train = 3, min_test = 3)
  # Works with ValidFraction
  @test train_test_counts(10, ValidFraction(0.7)) == (7, 3)
end
