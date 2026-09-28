function load_dataset(dir::AbstractString, structure::AbstractString, D::Int, N::Int)
  name = "$(structure)_D$(D)_N$(N)"
  rd(suffix, T, dims) = read!(joinpath(dir, "$(name)_$(suffix).bin"), Array{T}(undef, dims))

  return (
    name = name,
    structure = structure,
    D = D,
    N = N,
    X = rd("X", Float64, (D, N)),
    y = rd("y", Float64, N),
    times = rd("times", Int, N),
  )
end
