module DataSplits

export split
include("core.jl")
include("strategies/random.jl")
include("strategies/LazyKennardStone.jl")

end
