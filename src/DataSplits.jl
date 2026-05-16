module DataSplits

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
include("strategies/GroupKFold.jl")
include("strategies/KFold.jl")
include("strategies/LeavePOut.jl")
include("strategies/LeavePGroupsOut.jl")
include("strategies/StratifiedKFold.jl")
include("strategies/ShuffleSplit.jl")
include("strategies/StratifiedShuffleSplit.jl")
include("strategies/PredefinedSplit.jl")
include("strategies/BootstrapSplit.jl")
include("strategies/BlockedCV.jl")
include("strategies/RepeatedKFold.jl")
include("strategies/RepeatedStratifiedKFold.jl")
include("strategies/TimeSeriesSplit.jl")
include("strategies/StratifiedGroupKFold.jl")
include("strategies/PurgedKFold.jl")


# Core API
export partition
export AbstractSplitResult, AbstractSplitStrategy, AbstractCVStrategy
export AbstractResamplingCVStrategy
export splitdata, splitview
export trainview, testview, valview
export traindata, testdata, valdata
export trainindices, testindices, valindices, folds, rowpairs

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

# Cross-validation
export GroupKFold,
  KFold, LeavePOut, LeaveOneOut, LeavePGroupsOut, LeaveOneGroupOut, TimeSeriesSplit
export StratifiedKFold
export GroupKFold, KFold, LeavePOut, LeaveOneOut, LeavePGroupsOut, LeaveOneGroupOut
export StratifiedKFold, StratifiedGroupKFold
export ShuffleSplit, StratifiedShuffleSplit
export PredefinedSplit
export BootstrapSplit
export BlockedCV, PurgedKFold
export RepeatedKFold, RepeatedStratifiedKFold

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
