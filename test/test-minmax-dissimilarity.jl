@testset "Dissimilarity constructor validation" begin
  import DataSplits: SplitParameterError
  @test_throws SplitParameterError MaximumDissimilaritySplit(; distance_cutoff = -0.1)
  @test_throws SplitParameterError LazyMaximumDissimilaritySplit(; distance_cutoff = -0.1)
  @test_throws SplitParameterError LazyMinimumDissimilaritySplit(; distance_cutoff = -0.1)
  # MinimumDissimilaritySplit delegates to OptiSimSplit, so same check applies
  @test_throws SplitParameterError MinimumDissimilaritySplit(; distance_cutoff = -0.1)
end
