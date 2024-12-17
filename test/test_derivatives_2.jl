using Pkg
Pkg.activate("test/Project.toml")
using Test
using QPGreen
using Printf, LinearAlgebra
using BenchmarkTools: @btime

Z = (0.02π, 0.1);

params = (α=0.3, k=(√(10)), c=0.6, c̃=1.0, ε=0.4341, order=8);

res_eig_x1, res_eig_x2 = analytical_derivative(Z, params; nb_terms=1e6)

grid_size = 1024;
interp1, interp2, cache = fm_method_preparation_derivative(params, grid_size);
fm_method_calculation_derivative(Z, params, interp1, interp2, cache; nb_terms=32)

@time for i ∈ [32, 64, 128, 256, 512, 1024]
    interp1, interp2, cache = fm_method_preparation_derivative(params, i)
    res_fm = fm_method_calculation_derivative(Z, params, interp1, interp2, cache; nb_terms=32)
    str_err_1 = @sprintf "%.2E" abs(res_fm[1] - res_eig_x1)/abs(res_eig_x1)
    println("Grid size: ", i, " and res: ", res_fm[1], " and error: ", str_err_1)

    str_err_2 = @sprintf "%.2E" abs(res_fm[2] - res_eig_x2)/abs(res_eig_x2)
    println("Grid size: ", i, " and res: ", res_fm[2], " and error: ", str_err_2)
    println(" ")
end
