using Test
using DataSplits

# All AbstractSplitStrategy instances that produce TrainTestSplit — valid for
# the two-strategy validation-split API. Grouped times by 3 in auxiliary data
# ensure GroupShuffleSplit always overshoots (pool ≥ n_train+n_val), preventing
# BoundsError in the inner split.
const _vp_strats = [
  (RandomSplit(), "RandomSplit"),
  (KennardStoneSplit(), "KennardStoneSplit"),
  (LazyKennardStoneSplit(), "LazyKennardStoneSplit"),
  (OptiSimSplit(), "OptiSimSplit"),
  (LazyOptiSimSplit(), "LazyOptiSimSplit"),
  (MaximumDissimilaritySplit(), "MaximumDissimilaritySplit"),
  (LazyMaximumDissimilaritySplit(), "LazyMaximumDissimilaritySplit"),
  (MinimumDissimilaritySplit(), "MinimumDissimilaritySplit"),
  (LazyMinimumDissimilaritySplit(), "LazyMinimumDissimilaritySplit"),
  (MoraisLimaMartinSplit(), "MoraisLimaMartinSplit"),
  (MDKSSplit(), "MDKSSplit"),
  (LazyMDKSSplit(), "LazyMDKSSplit"),
  (SPXYSplit(), "SPXYSplit"),
  (LazySPXYSplit(), "LazySPXYSplit"),
  (TargetPropertyHigh(), "TargetPropertyHigh"),
  (TargetPropertyLow(), "TargetPropertyLow"),
  (TimeSplitOldest(), "TimeSplitOldest"),
  (TimeSplitNewest(), "TimeSplitNewest"),
  (GroupShuffleSplit(), "GroupShuffleSplit"),
  (GroupStratifiedSplit(:proportional), "GroupStratifiedSplit"),
]

const _vp_pairs = [(a1, a2, n1, n2) for (a1, n1) in _vp_strats for (a2, n2) in _vp_strats]
const _vp_n = length(_vp_pairs)   # 400

# Generator: pick a random (outer, inner) strategy pair and random cohort sizes.
# N ∈ [30, 50] keeps distance-based strategies fast; groups of 3 ensure the
# inner pool is always large enough for any inner strategy (GroupShuffleSplit
# always overshoots, so pool ≥ n_train_user + n_val_user).
const _vp_gen = @composed function make_vp_case(
  pair_idx = Data.Integers(1, _vp_n),
  N = Data.Integers(30, 50),
)
  n_test = Data.produce!(Data.Integers(1, N - 2))
  n_val = Data.produce!(Data.Integers(1, N - n_test - 1))
  n_train = N - n_test - n_val
  return (pair_idx, N, n_train, n_val, n_test)
end

@testset "Validation pairs" begin
  @check max_examples = 3000 rng = Xoshiro(80) function vp_all_pairs_valid(case = _vp_gen)
    pair_idx, N, n_train, n_val, n_test = case
    alg1, alg2, _, _ = _vp_pairs[pair_idx]
    X = reshape(collect(1.0:N), 1, N)
    all_slots = union(consumes(alg1), consumes(alg2))
    extra = Dict{Symbol,Any}()
    :target ∈ all_slots && (extra[:target] = collect(1.0:N))
    :time ∈ all_slots && (extra[:time] = collect(1:N))
    :groups ∈ all_slots && (extra[:groups] = [div(k - 1, 3) + 1 for k = 1:N])
    result = partition(
      X,
      alg1,
      alg2;
      train = n_train,
      validation = n_val,
      test = n_test,
      rng = Xoshiro(42),
      extra...,
    )
    is_full_partition(result, N) && cohorts_are_complements(result, N)
  end
end
