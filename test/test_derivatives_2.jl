using Pkg
Pkg.activate("test/Project.toml")
using Test
using QPGreen
using Printf, LinearAlgebra
using BenchmarkTools: @btime

Z = (0.002π, 0.0);
params = (α=0.3, k=1.0, c=0.6, c̃=1.0, ε=0.4341, order=8);

# @btime res_eig = eigfunc_expansion($Z, $params.k, $params.α; nb_terms=100);
# @btime res_img = image_expansion($Z, $params.k, $params.α; nb_terms=100);

# res_der = analytical_derivative(Z, params.k, params.α; nb_terms=100);
res = (0.050308789349485 + 0.10989902020665852im, 0.0 + 0.0im); # obtained by running the above code for nb_terms=800_000_000

grid_size = 128;
preparation_result, interp, cache = fm_method_preparation_derivative(params, grid_size);
fm_method_calculation_derivative(Z, params, preparation_result, interp, cache; nb_terms=32)

# res_eig = 0.7685069487 + 0.1952423542im
# @time for i ∈ [32, 64, 128, 256, 512, 1024]
#     preparation_result, interp, cache = QPGreen.fm_method_preparation(params, i)
#     res_fm = QPGreen.fm_method_calculation(Z, params, preparation_result, interp, cache; nb_terms=32)
#     str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
#     println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
# end
