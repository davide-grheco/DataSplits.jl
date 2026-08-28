using DataSplits
using Documenter
using DocumenterCitations
using DocumenterCodeBlocks

DocMeta.setdocmeta!(DataSplits, :DocTestSetup, :(using DataSplits); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
  file for file in readdir(joinpath(@__DIR__, "src")) if
  file != "index.md" && splitext(file)[2] == ".md"
]

bib = CitationBibliography("src/refs.bib"; style = :numeric)

makedocs(;
  modules = [DataSplits],
  authors = "Davide Crucitti <davide.crucitti@grheco.com>",
  repo = "https://github.com/davide-grheco/DataSplits.jl/blob/{commit}{path}#{line}",
  sitename = "DataSplits.jl",
  format = Documenter.HTML(; canonical = "https://davide-grheco.github.io/DataSplits.jl"),
  pages = ["index.md"; numbered_pages],
  plugins = [bib, CodeBlocks()],
)

deploydocs(; repo = "github.com/davide-grheco/DataSplits.jl")
