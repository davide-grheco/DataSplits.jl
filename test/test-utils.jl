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
# -----------------------------------------------------------------------
# Helpers: groupsortperm, group_offsets, distribute_blocks
# -----------------------------------------------------------------------

import DataSplits: groupsortperm, group_offsets, distribute_blocks

@testset "groupsortperm" begin
  @testset "basic" begin
    time = [3, 1, 2, 1, 3]
    sorted_keys, perm = groupsortperm(time)

    @test sorted_keys == [1, 2, 3]
    @test issorted(time[perm])
    @test sort(perm) == [1, 2, 3, 4, 5]
    # stable: equal keys appear in original relative order
    @test perm[1:2] == [2, 4]   # both obs with value 1, left-to-right
    @test perm[3:3] == [3]
    @test perm[4:5] == [1, 5]
  end

  @testset "all identical" begin
    time = [:a, :a, :a]
    sorted_keys, perm = groupsortperm(time)
    @test sorted_keys == [:a]
    @test sort(perm) == [1, 2, 3]
  end

  @testset "all unique" begin
    time = [10, 20, 30]
    sorted_keys, perm = groupsortperm(time)
    @test sorted_keys == [10, 20, 30]
    @test perm == [1, 2, 3]
  end

  @testset "reverse input" begin
    time = [3, 2, 1]
    sorted_keys, perm = groupsortperm(time)
    @test sorted_keys == [1, 2, 3]
    @test perm == [3, 2, 1]
  end
end

const time_vec_gen = @composed function make_time_vec(n_distinct = Data.Integers(1, 30))
  times = Int[]
  for t = 1:n_distinct
    reps = Data.produce!(Data.Integers(1, 6))
    append!(times, fill(t, reps))
  end
  shuffle!(Xoshiro(42), times)
  return times
end

@testset "groupsortperm (PBT)" begin
  @check max_examples = 500 rng = Xoshiro(10) function groupsortperm_perm_is_permutation(
    v = time_vec_gen,
  )
    N = length(v)
    _, perm = groupsortperm(v)
    sort(perm) == collect(1:N)
  end

  @check max_examples = 500 rng = Xoshiro(11) function groupsortperm_perm_sorts_v(
    v = time_vec_gen,
  )
    _, perm = groupsortperm(v)
    issorted(view(v, perm))
  end

  @check max_examples = 500 rng = Xoshiro(12) function groupsortperm_keys_are_sorted(
    v = time_vec_gen,
  )
    sorted_keys, _ = groupsortperm(v)
    issorted(sorted_keys)
  end

  @check max_examples = 500 rng = Xoshiro(13) function groupsortperm_key_count_equals_unique(
    v = time_vec_gen,
  )
    sorted_keys, _ = groupsortperm(v)
    length(sorted_keys) == length(unique(v))
  end

  @check max_examples = 500 rng = Xoshiro(14) function groupsortperm_keys_equal_sorted_unique(
    v = time_vec_gen,
  )
    sorted_keys, _ = groupsortperm(v)
    sorted_keys == sort(unique(v))
  end
end

@testset "group_offsets" begin
  @testset "basic" begin
    time = [3, 1, 2, 1, 3]
    sorted_keys, perm = groupsortperm(time)
    off = group_offsets(sorted_keys, perm, time)

    @test off == [0, 2, 3, 5]
    @test off[1] == 0
    @test off[end] == length(time)
    @test issorted(off)
    @test Set(perm[(off[1]+1):off[2]]) == Set([2, 4])   # key = 1
    @test Set(perm[(off[2]+1):off[3]]) == Set([3])       # key = 2
    @test Set(perm[(off[3]+1):off[4]]) == Set([1, 5])    # key = 3
  end

  @testset "single group" begin
    time = [7, 7, 7]
    sorted_keys, perm = groupsortperm(time)
    off = group_offsets(sorted_keys, perm, time)
    @test off == [0, 3]
  end

  @testset "all unique" begin
    time = [5, 3, 1]
    sorted_keys, perm = groupsortperm(time)
    off = group_offsets(sorted_keys, perm, time)
    @test off == [0, 1, 2, 3]
    @test perm[1] == 3   # value 1 at original index 3
    @test perm[2] == 2   # value 3 at original index 2
    @test perm[3] == 1   # value 5 at original index 1
  end
end

@testset "group_offsets (PBT)" begin
  @check max_examples = 500 rng = Xoshiro(20) function group_offsets_start_zero_end_N(
    v = time_vec_gen,
  )
    N = length(v)
    sk, perm = groupsortperm(v)
    off = group_offsets(sk, perm, v)
    off[1] == 0 && off[end] == N
  end

  @check max_examples = 500 rng = Xoshiro(21) function group_offsets_are_sorted(
    v = time_vec_gen,
  )
    sk, perm = groupsortperm(v)
    off = group_offsets(sk, perm, v)
    issorted(off)
  end

  @check max_examples = 500 rng = Xoshiro(22) function group_offsets_length_is_B_plus_1(
    v = time_vec_gen,
  )
    sk, perm = groupsortperm(v)
    off = group_offsets(sk, perm, v)
    length(off) == length(sk) + 1
  end

  @check max_examples = 500 rng = Xoshiro(23) function group_offsets_each_block_has_correct_key(
    v = time_vec_gen,
  )
    sk, perm = groupsortperm(v)
    off = group_offsets(sk, perm, v)
    all(enumerate(sk)) do (b, k)
      all((off[b]+1):off[b+1]) do pos
        v[perm[pos]] == k
      end
    end
  end

  @check max_examples = 500 rng = Xoshiro(24) function group_offsets_block_sizes_match_counts(
    v = time_vec_gen,
  )
    sk, perm = groupsortperm(v)
    off = group_offsets(sk, perm, v)
    all(enumerate(sk)) do (b, k)
      off[b+1] - off[b] == count(==(k), v)
    end
  end
end

# PBT generators for distribute_blocks
const distribute_gen =
  @composed function make_distribute_case(n_chunks = Data.Integers(1, 20))
    B = Data.produce!(Data.Integers(n_chunks, n_chunks + 40))
    return (B, n_chunks)
  end

@testset "distribute_blocks" begin
  @testset "exact division" begin
    ends = distribute_blocks(12, 4)
    @test ends == [3, 6, 9, 12]
  end

  @testset "remainder distributed to early chunks" begin
    ends = distribute_blocks(10, 3)
    sizes = diff([0; ends])
    @test sizes == [4, 3, 3]
  end

  @testset "single chunk" begin
    @test distribute_blocks(7, 1) == [7]
  end

  @testset "B equals n_chunks" begin
    ends = distribute_blocks(5, 5)
    @test ends == [1, 2, 3, 4, 5]
  end
end

@testset "distribute_blocks (PBT)" begin
  @check max_examples = 500 rng = Xoshiro(30) function distribute_ends_at_B(
    case = distribute_gen,
  )
    B, n_chunks = case
    distribute_blocks(B, n_chunks)[end] == B
  end

  @check max_examples = 500 rng = Xoshiro(31) function distribute_is_strictly_increasing(
    case = distribute_gen,
  )
    B, n_chunks = case
    ends = distribute_blocks(B, n_chunks)
    all(i -> ends[i] < ends[i+1], 1:(length(ends)-1))
  end

  @check max_examples = 500 rng = Xoshiro(32) function distribute_sizes_differ_by_at_most_one(
    case = distribute_gen,
  )
    B, n_chunks = case
    ends = distribute_blocks(B, n_chunks)
    sizes = diff([0; ends])
    maximum(sizes) - minimum(sizes) <= 1
  end

  @check max_examples = 500 rng = Xoshiro(33) function distribute_larger_chunks_come_first(
    case = distribute_gen,
  )
    B, n_chunks = case
    ends = distribute_blocks(B, n_chunks)
    sizes = diff([0; ends])
    issorted(sizes; rev = true)
  end

  @check max_examples = 500 rng = Xoshiro(34) function distribute_total_equals_B(
    case = distribute_gen,
  )
    B, n_chunks = case
    ends = distribute_blocks(B, n_chunks)
    sizes = diff([0; ends])
    sum(sizes) == B
  end
end
