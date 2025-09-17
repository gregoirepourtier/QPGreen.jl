using Pkg
Pkg.activate("test/Project.toml")
using Test
using QPGreen
using Printf, LinearAlgebra
using BenchmarkTools: @btime

Z = (0.02π, 0.1); # Z = (0.02π, 0.1);
params = (alpha=0.3, k=(√(10)), c=0.6, c_tilde=1.0, epsilon=0.4341, order=8);

res_img = QPGreen.image_expansion_hessian(Z, params; nb_terms=1e7)
res_eig = QPGreen.eigfunc_expansion_hessian(Z, params; period=2π, nb_terms=1e7)

# res_img = QPGreen.image_expansion_gradient(Z, params; nb_terms=1e6)
# res_eig = QPGreen.eigfunc_expansion_gradient(Z, params; period=2π, nb_terms=1e6)



####
grid_size = 1024;
value, grad, hess, cache = init_qp_green_fft_mod(params, grid_size; gradient=true, hessian=true);
hess_eval = hess_qp_green_mod(Z, params, hess, cache; nb_terms=32)

@time for i ∈ [32, 64, 128, 256, 512, 1024]
    value, grad, hess, cache = init_qp_green_fft_mod(params, i; gradient=true, hessian=true)
    res_fm = hess_qp_green_mod(Z, params, hess, cache; nb_terms=32)
    str_err_1 = @sprintf "%.2E" abs(res_fm[1] - res_eig[1, 1])/abs(res_eig[1])
    println("Grid size: ", i, " and res: ", res_fm[1], " and error: ", str_err_1)

    str_err_2 = @sprintf "%.2E" abs(res_fm[2] - res_eig[1, 2])/abs(res_eig[2])
    println("Grid size: ", i, " and res: ", res_fm[2], " and error: ", str_err_2)

    str_err_3 = @sprintf "%.2E" abs(res_fm[3] - res_eig[2, 2])/abs(res_eig[3])
    println("Grid size: ", i, " and res: ", res_fm[3], " and error: ", str_err_3)
    println(" ")
end
