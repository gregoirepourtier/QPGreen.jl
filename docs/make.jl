using QPGreen
using Documenter
using DocumenterCitations

# from https://github.com/IntegralEquations/Inti.jl/blob/main/docs/make.jl &  https://github.com/fonsp/Pluto.jl/pull/2471
function generate_plaintext(notebook,
                            strmacrotrim::Union{String, Nothing}=nothing;
                            header::Function=_ -> nothing,
                            footer::Function=_ -> nothing,
                            textcomment::Function=identity,
                            codewrapper::Function,)
    cell_strings = String[]
    header_content = header(notebook)
    isnothing(header_content) || push!(cell_strings, header_content)
    for cell_id ∈ notebook.cell_order
        cell = notebook.cells_dict[cell_id]
        scode = strip(cell.code)
        (raw, ltrim, rtrim) = if isnothing(strmacrotrim)
            false, 0, 0
        elseif startswith(scode, string(strmacrotrim, '"'^3))
            true, length(strmacrotrim) + 3, 3
        elseif startswith(scode, string(strmacrotrim, '"'))
            true, length(strmacrotrim) + 1, 1
        else
            false, 0, 0
        end
        push!(cell_strings,
              if raw
                  text = strip(scode[nextind(scode, 1, ltrim):prevind(scode, end, rtrim)],
                               ['\n'])
                  ifelse(Pluto.is_disabled(cell), textcomment, identity)(text)
              else
                  codewrapper(cell, Pluto.is_disabled(cell))
              end)
    end
    footer_content = footer(notebook)
    isnothing(footer_content) || push!(cell_strings, footer_content)
    return join(cell_strings, "\n\n")
end

function generate_md(input; output=replace(input, r"\.jl$" => ".md"))
    fname = basename(input)
    notebook = Pluto.load_notebook(input)
    header = _ -> "[![Pluto notebook](https://img.shields.io/badge/download-Pluto_notebook-blue)]($fname)"

    function codewrapper(cell, _)
        # 1. Strips begin/end block
        # 2. Reformats code using JuliaFormatter
        # 3. Wraps all code in same ```@example``` block for documenter
        code = strip(cell.code)
        if startswith(code, "begin") && endswith(code, "end")
            code = strip(code[6:(end - 4)])  # Remove "begin" and "end" and strip spaces
            # reformat code using JuliaFormatter
            code = format_text(String(code))
        end
        return if cell.code_folded
            string("```@setup $fname\n", code, "\n```")
        else
            string("```@example $fname\n", code, "\n```")
        end
    end
    textcomment(text) = string("<!-- ", text, " -->")
    str = generate_plaintext(notebook, "md"; header, codewrapper, textcomment)

    open(output, "w") do io
        return write(io, str)
    end
    return output
end



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
