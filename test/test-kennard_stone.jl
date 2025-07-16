@testset "LazyKennardStone (CADEX)" begin
  using Random
  using Distances
  using DataSplits
  using Test

  function check_split(train, test, N; n_test_expected = nothing)
    @test sort(vcat(train, test)) == 1:N
    @test isempty(intersect(train, test))
    if n_test_expected !== nothing
      @test length(test) == n_test_expected
      @test length(train) == N - length(test)
    end
  end

  Random.seed!(42)

  # Test 1: small numeric dataset (matrix)
  X = rand(50, 3)
  strat = LazyKennardStoneSplit(0.8, Euclidean())
  tr, te = DataSplits.split(X, strat; rng = Random.GLOBAL_RNG)
  check_split(tr, te, 50; n_test_expected = 10)

  # Test 2: vector-of-vectors input
  Xv = [X[i, :] for i = 1:size(X, 1)]
  tr2, te2 =
    DataSplits.split(Xv, LazyKennardStoneSplit(0.3, CosineDist()); rng = Random.GLOBAL_RNG)
  check_split(tr2, te2, 50; n_test_expected = round(Int, 0.7 * 50))

  # Test 3: deterministic behavior with fixed RNG
  rng1 = MersenneTwister(123)
  tr1, te1 = DataSplits.split(X, LazyKennardStoneSplit(0.25, Euclidean()); rng = rng1)
  rng2 = MersenneTwister(123)
  tr2b, te2b = DataSplits.split(X, LazyKennardStoneSplit(0.25, Euclidean()); rng = rng2)
  @test tr1 == tr2b && te1 == te2b

  # Test 4: frac boundary handling
  stratf = LazyKennardStoneSplit(0.02, Euclidean())
  @test_throws ArgumentError trf, tef = DataSplits.split(rand(10, 2), stratf)

  stratf = LazyKennardStoneSplit(0.98, Euclidean())
  @test_throws ArgumentError trf, tef = DataSplits.split(rand(10, 2), stratf)

  # Test 5: edge-case of too-small frac
  @test_throws ArgumentError LazyKennardStoneSplit(0.0, Euclidean())
  @test_throws ArgumentError DataSplits.split(
    rand(5, 5),
    LazyKennardStoneSplit(1.0, Euclidean()),
  )

  # Test 6: expected max-min property (somewhat)
  X2 = vcat(randn(5, 2) .+ 5, randn(5, 2) .- 5)
  tr_small, te_small = DataSplits.split(X2, LazyKennardStoneSplit(0.2, Euclidean()))
  # Ensure one point from each cluster is selected
  labels = [x[1] > 0 ? 1 : -1 for x in eachrow(X2)]
  test_labels = labels[te_small]
  @test in(1, test_labels) && in(-1, test_labels)
end
