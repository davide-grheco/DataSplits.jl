using Random, Printf, SHA, TOML

const CFG = TOML.parsefile(joinpath(@__DIR__, "manifest.toml"))
const OUTDIR = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "data")

function standardise!(X)
  for i in axes(X, 1)
    row = @view X[i, :]
    m = sum(row) / length(row)
    row .-= m
    s = sqrt(sum(abs2, row) / max(length(row) - 1, 1))
    s > 0 && (row ./= s)
  end
  return X
end

function generate(D::Int, N::Int, rng)
  X = randn(rng, D, N)
  return standardise!(X)
end

function metadata(N::Int, rng)
  return (y = randn(rng, N), times = collect(1:N))
end

dump_array(path, a) = open(io -> write(io, a), path, "w")

function main()
  mkpath(OUTDIR)
  seed = 20260922
  index = IOBuffer()
  println(index, "# name\tstructure\tD\tN\tsha256(X)")

  cells =
    [("isotropic", Int(CFG["data"]["features"]), N) for N in Int.(CFG["data"]["sizes"])]

  for (structure, D, N) in cells
    rng = Xoshiro(seed + hash((structure, D, N)) % 100_000)
    X = generate(D, N, rng)
    md = metadata(N, rng)

    name = "$(structure)_D$(D)_N$(N)"
    dump_array(joinpath(OUTDIR, "$(name)_X.bin"), X)
    dump_array(joinpath(OUTDIR, "$(name)_y.bin"), md.y)
    dump_array(joinpath(OUTDIR, "$(name)_times.bin"), md.times)

    digest = bytes2hex(sha256(reinterpret(UInt8, vec(X))))
    println(index, "$name\t$structure\t$D\t$N\t$digest")
    @printf(
      "%-28s %s x %-7s  %6.1f MiB  %s\n",
      name,
      D,
      N,
      sizeof(X) / 2^20,
      first(digest, 16)
    )
  end

  write(joinpath(OUTDIR, "index.tsv"), String(take!(index)))
  println("\nwrote ", length(cells), " datasets to ", OUTDIR)
end

main()
