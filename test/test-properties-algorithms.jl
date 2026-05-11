using DataSplits:
  partition,
  KennardStoneSplit,
  LazyKennardStoneSplit,
  SPXYSplit,
  LazySPXYSplit,
  MDKSSplit,
  LazyMDKSSplit,
  OptiSimSplit,
  LazyOptiSimSplit,
  MaximumDissimilaritySplit,
  LazyMaximumDissimilaritySplit,
  MinimumDissimilaritySplit,
  LazyMinimumDissimilaritySplit

const algo_sizes_gen = @composed function make_algo_sizes(N = Data.Integers(3, 50))
  n_train = Data.produce!(Data.Integers(1, N - 1))
  return (N, n_train, N - n_train)
end

@testset "KennardStone algorithm properties" begin
  @check max_examples = 200 rng = Xoshiro(60) function ks_is_full_partition(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    result = partition(X, KennardStoneSplit(); train = n_train, test = n_test)
    is_full_partition(result, N) &&
      has_correct_split_size(result, n_train, n_test) &&
      cohorts_are_complements(result, N)
  end

  @check max_examples = 200 rng = Xoshiro(61) function ks_lazy_eager_agree(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    r_eager = partition(X, KennardStoneSplit(); train = n_train, test = n_test)
    r_lazy = partition(X, LazyKennardStoneSplit(); train = n_train, test = n_test)
    Set(trainindices(r_eager)) == Set(trainindices(r_lazy))
  end
end

@testset "SPXY algorithm properties" begin
  @check max_examples = 200 rng = Xoshiro(62) function spxy_is_full_partition(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    y = collect(1.0:N)
    result = partition(X, SPXYSplit(); target = y, train = n_train, test = n_test)
    is_full_partition(result, N) &&
      has_correct_split_size(result, n_train, n_test) &&
      cohorts_are_complements(result, N)
  end

  @check max_examples = 200 rng = Xoshiro(63) function spxy_lazy_eager_agree(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    y = collect(1.0:N)
    r_eager = partition(X, SPXYSplit(); target = y, train = n_train, test = n_test)
    r_lazy = partition(X, LazySPXYSplit(); target = y, train = n_train, test = n_test)
    Set(trainindices(r_eager)) == Set(trainindices(r_lazy))
  end
end

@testset "MDKS algorithm properties" begin
  @check max_examples = 200 rng = Xoshiro(64) function mdks_is_full_partition(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    y = collect(1.0:N)
    result = partition(X, MDKSSplit(); target = y, train = n_train, test = n_test)
    is_full_partition(result, N) &&
      has_correct_split_size(result, n_train, n_test) &&
      cohorts_are_complements(result, N)
  end

end

@testset "OptiSim algorithm properties" begin
  @check max_examples = 100 rng = Xoshiro(66) function optisim_is_full_partition(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    result = partition(X, OptiSimSplit(); train = n_train, test = n_test, rng = Xoshiro(42))
    is_full_partition(result, N) && cohorts_are_complements(result, N)
  end

  @check max_examples = 100 rng = Xoshiro(67) function optisim_lazy_eager_agree(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    r_eager =
      partition(X, OptiSimSplit(); train = n_train, test = n_test, rng = Xoshiro(42))
    r_lazy =
      partition(X, LazyOptiSimSplit(); train = n_train, test = n_test, rng = Xoshiro(42))
    Set(trainindices(r_eager)) == Set(trainindices(r_lazy))
  end
end

@testset "MaximumDissimilarity algorithm properties" begin
  @check max_examples = 100 rng = Xoshiro(68) function maxdissim_is_full_partition(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    result = partition(
      X,
      MaximumDissimilaritySplit();
      train = n_train,
      test = n_test,
      rng = Xoshiro(42),
    )
    is_full_partition(result, N) && cohorts_are_complements(result, N)
  end

  @check max_examples = 100 rng = Xoshiro(69) function maxdissim_lazy_eager_agree(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    r_eager = partition(
      X,
      MaximumDissimilaritySplit();
      train = n_train,
      test = n_test,
      rng = Xoshiro(42),
    )
    r_lazy = partition(
      X,
      LazyMaximumDissimilaritySplit();
      train = n_train,
      test = n_test,
      rng = Xoshiro(42),
    )
    Set(trainindices(r_eager)) == Set(trainindices(r_lazy))
  end
end

@testset "MinimumDissimilarity algorithm properties" begin
  @check max_examples = 100 rng = Xoshiro(71) function mindissim_is_full_partition(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    result = partition(
      X,
      MinimumDissimilaritySplit();
      train = n_train,
      test = n_test,
      rng = Xoshiro(42),
    )
    is_full_partition(result, N) && cohorts_are_complements(result, N)
  end

  @check max_examples = 100 rng = Xoshiro(72) function mindissim_lazy_eager_agree(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    r_eager = partition(
      X,
      MinimumDissimilaritySplit();
      train = n_train,
      test = n_test,
      rng = Xoshiro(42),
    )
    r_lazy = partition(
      X,
      LazyMinimumDissimilaritySplit();
      train = n_train,
      test = n_test,
      rng = Xoshiro(42),
    )
    Set(trainindices(r_eager)) == Set(trainindices(r_lazy))
  end
end
