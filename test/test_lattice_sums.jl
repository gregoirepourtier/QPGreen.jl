using Pkg

Pkg.activate("test/Project.toml")

using Test
using GreenFunction


function test_lattice_sum(x)

    d, α, k = (2 * π, 0.3, 10.0)

end


x = SVector(10.0, 0.4)
@time test_lattice_sum(x)

GreenFunction.green_function_eigfct_exp(x; nb_terms=1000)
GreenFunction.green_function_img_exp(x; nb_terms=500000)
