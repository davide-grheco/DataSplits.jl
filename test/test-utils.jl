using Test
using DataSplits: SplitParameterError
import DataSplits: ValidFraction

@testset "_resolve_sizes" begin
  # Percentages (sum to 100)
  @test DataSplits._resolve_sizes(10, 70, nothing, 30) == (7, 0, 3)
  @test DataSplits._resolve_sizes(10, 30, nothing, 70) == (3, 0, 7)
  @test DataSplits._resolve_sizes(200, 80, nothing, 20) == (160, 0, 40)
  @test DataSplits._resolve_sizes(100, 70, 10, 20) == (70, 10, 20)
  @test DataSplits._resolve_sizes(200, 70, 10, 20) == (140, 20, 40)

  # Absolute counts (sum to N)
  @test DataSplits._resolve_sizes(10, 7, nothing, 3) == (7, 0, 3)
  @test DataSplits._resolve_sizes(50, 35, nothing, 15) == (35, 0, 15)
  @test DataSplits._resolve_sizes(50, 30, 10, 10) == (30, 10, 10)

  # Rounding remainder absorbed by train (50% of 7 → train=4, test=3)
  @test DataSplits._resolve_sizes(7, 50, nothing, 50) == (4, 0, 3)

  # Each value must be ≥ 1
  @test_throws SplitParameterError DataSplits._resolve_sizes(10, 0, nothing, 100)
  @test_throws SplitParameterError DataSplits._resolve_sizes(10, 100, nothing, 0)
  @test_throws SplitParameterError DataSplits._resolve_sizes(10, 70, 0, 30)

  # Sum is neither 100 nor N
  @test_throws SplitParameterError DataSplits._resolve_sizes(50, 30, nothing, 25)
  @test_throws SplitParameterError DataSplits._resolve_sizes(40, 30, nothing, 30)

  # Resolved n_train < 1 (1% of 10 = 0)
  @test_throws SplitParameterError DataSplits._resolve_sizes(10, 1, nothing, 99)
end
@testset "_resolve_sizes (fractions)" begin
  @test DataSplits._resolve_sizes(10, 0.7, nothing, 0.3) == (7, 0, 3)
  @test DataSplits._resolve_sizes(100, 0.7, 0.1, 0.2) == (70, 10, 20)
  @test DataSplits._resolve_sizes(200, 0.8, nothing, 0.2) == (160, 0, 40)

  # Floating-point imprecision tolerated
  @test DataSplits._resolve_sizes(10, 0.1 + 0.2, nothing, 0.7) == (3, 0, 7)

  # Out-of-range fractions
  @test_throws SplitParameterError DataSplits._resolve_sizes(10, 0.0, nothing, 1.0)
  @test_throws SplitParameterError DataSplits._resolve_sizes(10, 1.0, nothing, 0.0)

  # Sum ≠ 1
  @test_throws SplitParameterError DataSplits._resolve_sizes(10, 0.5, nothing, 0.6)

  # Resolved cohort < 1 (0.01 * 10 rounds to 0)
  @test_throws SplitParameterError DataSplits._resolve_sizes(10, 0.01, nothing, 0.99)
end

@testset "ValidFraction" begin
  vf = ValidFraction(0.8)
  @test vf.frac == 0.8
  @test vf * 10 ≈ 8.0
  @test 10 * vf ≈ 8.0
  @test float(vf) == 0.8
  @test convert(Float64, vf) == 0.8
  @test_throws SplitParameterError ValidFraction(0.0)
  @test_throws SplitParameterError ValidFraction(1.0)
  @test_throws SplitParameterError ValidFraction(-0.1)
  @test_throws SplitParameterError ValidFraction(1.5)
end
