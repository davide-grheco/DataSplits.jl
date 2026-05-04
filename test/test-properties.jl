using Test
using Random
using Supposition
using Supposition.Data
using DataSplits:
  partition, RandomSplit, TimeSplitOldest, TimeSplitNewest, GroupShuffleSplit

const abs_split_sizes_gen =
  @composed function make_abs_split_sizes(N = Data.Integers(2, 200))
    n_train = Data.produce!(Data.Integers(1, N - 1))
    return (N, n_train, N - n_train)
  end

@testset "Property tests" begin
  @testset "RandomSplit" begin
    @check max_examples = 500 rng = Xoshiro(1) function randomsplit_is_a_partition(
      case = abs_split_sizes_gen,
    )
      N, n_train, n_test = case
      X = reshape(collect(1:N), 1, N)

      result =
        partition(X, RandomSplit(); train = n_train, test = n_test, rng = Xoshiro(42))

      train = result.train
      test = result.test
      assigned = sort(vcat(train, test))

      length(train) == n_train &&
        length(test) == n_test &&
        isempty(intersect(train, test)) &&
        assigned == collect(1:N)
    end
  end

  @testset "TimeSplit" begin
    @check max_examples = 300 rng = Xoshiro(2) function oldest_puts_train_before_test(
      case = abs_split_sizes_gen,
    )
      N, n_train, n_test = case
      t = Date(2020, 1, 1) .+ Day.(0:(N-1))

      result = partition(t, TimeSplitOldest(); train = n_train, test = n_test)

      train = result.train
      test = result.test
      assigned = sort(vcat(train, test))

      length(train) == n_train &&
        length(test) == n_test &&
        isempty(intersect(train, test)) &&
        assigned == collect(1:N) &&
        all(t[i] <= t[j] for i in train, j in test)
    end

    @check max_examples = 300 rng = Xoshiro(3) function newest_puts_train_after_test(
      case = abs_split_sizes_gen,
    )
      N, n_train, n_test = case
      t = Date(2020, 1, 1) .+ Day.(0:(N-1))

      result = partition(t, TimeSplitNewest(); train = n_train, test = n_test)

      train = result.train
      test = result.test
      assigned = sort(vcat(train, test))

      length(train) == n_train &&
        length(test) == n_test &&
        isempty(intersect(train, test)) &&
        assigned == collect(1:N) &&
        all(t[i] >= t[j] for i in train, j in test)
    end
  end

  @testset "GroupShuffleSplit" begin
    @check max_examples = 300 rng = Xoshiro(4) function group_shuffle_does_not_split_groups(
      n_groups = Data.Integers(2, 30),
      group_size = Data.Integers(1, 20),
    )
      groups = repeat(1:n_groups; inner = group_size)

      result =
        partition(groups, GroupShuffleSplit(); train = 60, test = 40, rng = Xoshiro(42))

      train = result.train
      test = result.test
      assigned = sort(vcat(train, test))

      is_valid_partition =
        isempty(intersect(train, test)) && assigned == collect(eachindex(groups))

      groups_are_not_split = all(unique(groups)) do gid
        idxs = findall(==(gid), groups)
        in_train = any(in(train), idxs)
        in_test = any(in(test), idxs)
        !(in_train && in_test)
      end

      is_valid_partition && groups_are_not_split
    end
  end
end
