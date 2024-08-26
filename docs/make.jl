using GreenFunction
using Documenter

DocMeta.setdocmeta!(GreenFunction, :DocTestSetup, :(using GreenFunction); recursive=true)

makedocs(;
    modules=[GreenFunction],
    authors="gregoirepourtier <gpourtier@icloud.com> and contributors",
    sitename="GreenFunction.jl",
    format=Documenter.HTML(;
        canonical="https://gregoirepourtier.github.io/GreenFunction.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gregoirepourtier/GreenFunction.jl",
    devbranch="main",
)
