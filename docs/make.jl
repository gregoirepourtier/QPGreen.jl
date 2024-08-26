using MyPkg
using Documenter

DocMeta.setdocmeta!(MyPkg, :DocTestSetup, :(using MyPkg); recursive=true)

makedocs(;
    modules=[MyPkg],
    authors="gregoirepourtier <gpourtier@icloud.com> and contributors",
    sitename="MyPkg.jl",
    format=Documenter.HTML(;
        canonical="https://gregoirepourtier.github.io/MyPkg.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gregoirepourtier/MyPkg.jl",
    devbranch="main",
)
