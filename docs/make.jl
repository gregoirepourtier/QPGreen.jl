using QPGreen
using Documenter
using DocumenterCitations
using DocumenterInterLinks

DocMeta.setdocmeta!(QPGreen, :DocTestSetup, :(using QPGreen); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

makedocs(;
         modules=[QPGreen],
         authors="Grégoire Pourtier and Ruming Zhang",
         sitename="QPGreen.jl",
         format=Documenter.HTML(;
                                canonical="https://gregoirepourtier.github.io/QPGreen.jl",
                                edit_link="main",
                                assets=String[],),
         pages=["Home" => "index.md",
             "Examples" => "examples.md",
             "References" => "references.md",
             "Docstrings" => "docstrings.md"],
         plugins=[bib],)

deploydocs(;
           repo="github.com/gregoirepourtier/QPGreen.jl",
           devbranch="main",)
