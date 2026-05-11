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

# N > 100 guarantees sum = N ≠ 100, so the absolute-count branch always fires.
const resolve_abs_gen = @composed function make_resolve_abs(N = Data.Integers(101, 300))
  n_train = Data.produce!(Data.Integers(1, N - 1))
  return (N, n_train)
end

const fractional_threeway_size_case_gen =
  @composed function make_fractional_threeway_size_case(N = Data.Integers(3, 300))
    n_train = Data.produce!(Data.Integers(1, N - 2))
    n_val = Data.produce!(Data.Integers(1, N - n_train - 1))
    n_test = N - n_train - n_val

    train = n_train / N
    validation = n_val / N
    test = n_test / N

    return (N, train, validation, test)
  end

@testset "_resolve_sizes (PBT two-way)" begin
  @check max_examples = 300 rng = Xoshiro(94) function resolve_sizes_absolute_is_identity(
    case = resolve_abs_gen,
  )
    N, n_train = case
    n_test = N - n_train
    DataSplits._resolve_sizes(N, n_train, nothing, n_test) == (n_train, 0, n_test)
  end

  @check max_examples = 300 rng = Xoshiro(95) function resolve_sizes_percentage_sums_to_N(
    N = Data.Integers(10, 2000),
    train_pct = Data.Integers(1, 99),
  )
    test_pct = 100 - train_pct
    n_train, _, n_test = DataSplits._resolve_sizes(N, train_pct, nothing, test_pct)
    n_train + n_test == N && n_train >= 1 && n_test >= 1
  end
end

@testset "_resolve_sizes fractional three-way properties" begin
  @check max_examples = 300 rng = Xoshiro(95) function fractional_sizes_are_valid(
    case = fractional_threeway_size_case_gen,
  )
    N, train, validation, test = case
    n_train, n_val, n_test = DataSplits._resolve_sizes(N, train, validation, test)

    return n_train + n_val + n_test == N && n_train >= 1 && n_val >= 1 && n_test >= 1
  end
end
