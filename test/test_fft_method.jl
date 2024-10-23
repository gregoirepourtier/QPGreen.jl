using Pkg
Pkg.activate("test/Project.toml")
using Test, QPGreen, Printf, LinearAlgebra
using BenchmarkTools: @btime

Z = (0.002π, 0.0);
params = (α=0.3, k=1.0, c=0.6, c̃=1.0, ε=0.4341, order=8);

res_eig = eigfunc_expansion(Z, params.k, params.α; nb_terms=10000);
res_img = image_expansion(Z, params.k, params.α; nb_terms=200000);

grid_size = 5;
# @code_warntype fm_method_preparation(params, grid_size);
preparation_result, interp, cache = fm_method_preparation(params, grid_size);
fm_method_calculation(Z, params, preparation_result, interp, cache; nb_terms=32);

res_eig = 0.7685069487 + 0.1952423542im
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    preparation_result, interp, cache = QPGreen.fm_method_preparation(params, i)
    res_fm = QPGreen.fm_method_calculation(Z, params, preparation_result, interp, cache; nb_terms=32)
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end
