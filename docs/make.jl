using QPGreen
using Documenter

DocMeta.setdocmeta!(QPGreen, :DocTestSetup, :(using QPGreen); recursive=true)

makedocs(;
         modules=[QPGreen],
         authors="Grégoire Pourtier and Ruming Zhang",
         sitename="QPGreen.jl",
         format=Documenter.HTML(;
                                canonical="https://gregoirepourtier.github.io/QPGreen.jl",
                                edit_link="main",
                                assets=String[],),
         pages=["Home" => "index.md",
             "Docstrings" => "docstrings.md"],)

deploydocs(;
           repo="github.com/gregoirepourtier/QPGreen.jl",
           devbranch="main",)
