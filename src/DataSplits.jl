module DataSplits

include("core.jl")
include("utils.jl")
include("validation.jl")
include("core.jl")
include("strategies/random.jl")
include("strategies/LazyKennardStone.jl")
include("strategies/KennardStone.jl")
include("strategies/SPXY.jl")
include("strategies/LazySPXY.jl")
include("strategies/MoraisLimaMartinSplit.jl")
include("strategies/OptiSim.jl")
include("strategies/LazyOptiSim.jl")
include("strategies/MinimumDissimilarity.jl")
include("strategies/MaximumDissimilarity.jl")
include("strategies/GroupShuffleSplit.jl")
include("strategies/GroupStratifiedSplit.jl")
include("clustering/SphereExclusion.jl")
include("strategies/TargetProperty.jl")
include("strategies/TimeSplit.jl")

# Core API
export partition
export AbstractSplitResult, AbstractSplitStrategy
export splitdata, splitview
export trainindices, testindices, valindices, folds

# Trait interface (for custom strategy authors)
export consumes, fallback_from_data

# Distance-based strategies
export KennardStoneSplit, CADEXSplit
export LazyKennardStoneSplit, LazyCADEXSplit
export MoraisLimaMartinSplit
export SPXYSplit, MDKSSplit
export LazySPXYSplit, LazyMDKSSplit
export OptiSimSplit
export LazyOptiSimSplit
export MinimumDissimilaritySplit, LazyMinimumDissimilaritySplit
export MaximumDissimilaritySplit, LazyMaximumDissimilaritySplit

# Random
export RandomSplit

# Group-aware
export GroupShuffleSplit, GroupStratifiedSplit

# Target / time property
export TargetPropertySplit, TargetPropertyHigh, TargetPropertyLow
export TimeSplit, TimeSplitOldest, TimeSplitNewest

# Clustering utility
export sphere_exclusion

# Exceptions
export SplitInputError, SplitParameterError, SplitNotImplementedError

# Result types (for dispatch on custom splitdata methods)
export TrainTestSplit, TrainValTestSplit, CrossValidationSplit

end
