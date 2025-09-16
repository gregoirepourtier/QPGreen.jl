using Pkg
Pkg.activate("test/Project.toml")
using Test
using QPGreen
using Printf, LinearAlgebra
using BenchmarkTools: @btime

Z = (0.02π, 0.1);

params = (alpha=0.3, k=(√(10)), c=0.6, c_tilde=1.0, epsilon=0.4341, order=8);

res_eig = eigfunc_expansion_derivative(Z, params; nb_terms=1e6)

grid_size = 1024;
value, grad, cache = init_qp_green_fft(params, grid_size; derivative=true);
grad_G = grad_qp_green(Z, params, grad, cache; nb_terms=32);

@time for i ∈ [32, 64, 128, 256, 512, 1024]
    value, grad, cache = init_qp_green_fft(params, i; derivative=true)
    res_fm = grad_qp_green(Z, params, grad, cache; nb_terms=32)
    str_err_1 = @sprintf "%.2E" abs(res_fm[1] - res_eig[1])/abs(res_eig[1])
    println("Grid size: ", i, " and res: ", res_fm[1], " and error: ", str_err_1)

    str_err_2 = @sprintf "%.2E" abs(res_fm[2] - res_eig[2])/abs(res_eig[2])
    println("Grid size: ", i, " and res: ", res_fm[2], " and error: ", str_err_2)
    println(" ")
end

## modified API
grid_size = 1024;
value, grad, cache = init_qp_green_fft_mod(params, grid_size; derivative=true);
grad_G = grad_qp_green_mod(Z, params, grad, cache; nb_terms=32)

@time for i ∈ [32, 64, 128, 256, 512, 1024]
    value, grad, cache = init_qp_green_fft_mod(params, i; derivative=true)
    res_fm = grad_qp_green_mod(Z, params, grad, cache; nb_terms=32)
    str_err_1 = @sprintf "%.2E" abs(res_fm[1] - res_eig[1])/abs(res_eig[1])
    println("Grid size: ", i, " and res: ", res_fm[1], " and error: ", str_err_1)

    str_err_2 = @sprintf "%.2E" abs(res_fm[2] - res_eig[2])/abs(res_eig[2])
    println("Grid size: ", i, " and res: ", res_fm[2], " and error: ", str_err_2)
    println(" ")
end