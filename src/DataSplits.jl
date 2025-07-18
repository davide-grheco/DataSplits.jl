module DataSplits

include("core.jl")
include("utils.jl")
include("strategies/random.jl")
include("strategies/LazyKennardStone.jl")
include("strategies/KennardStone.jl")
include("strategies/SPXY.jl")
include("strategies/OptiSim.jl")
include("strategies/ClusterShuffleSplit.jl")
include("clustering/SphereExclusion.jl")

export split
export SplitStrategy
export LazyCADEXSplit, LazyKennardStoneSplit, KennardStoneSplit
export SPXYSplit
export OptiSimSplit
export RandomSplit
export ClusterShuffleSplit
export sphere_exclusion

end
