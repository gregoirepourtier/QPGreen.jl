using Pkg
Pkg.activate("test/Project.toml")
using Test
using QPGreen, Bessels
using Printf, LinearAlgebra
using BenchmarkTools: @btime

Z = (0.02π, 0.1);
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


## Smooth
res_img_smooth = QPGreen.image_expansion_hessian_smooth(Z, params; nb_terms=1e7)
singularity = [-im / 4 * params.k^2 *
               (Bessels.hankelh1(1, params.k * norm(Z)) / (params.k * norm(Z)) - Bessels.hankelh1(2, params.k * norm(Z))) *
               Z[1]^2 / (norm(Z)^2)-im / 4 * params.k * Bessels.hankelh1(1, params.k * norm(Z)) * (1 / norm(Z) - Z[1]^2 / norm(Z)^3);
               -im / 4 * params.k^2 *
               (Bessels.hankelh1(1, params.k * norm(Z)) / (params.k * norm(Z)) - Bessels.hankelh1(2, params.k * norm(Z))) * Z[1] *
               Z[2] / (norm(Z)^2)-im / 4 * params.k * Bessels.hankelh1(1, params.k * norm(Z)) * (-Z[1] * Z[2] / norm(Z)^3);;
               -im / 4 * params.k^2 *
               (Bessels.hankelh1(1, params.k * norm(Z)) / (params.k * norm(Z)) - Bessels.hankelh1(2, params.k * norm(Z))) * Z[1] *
               Z[2] / (norm(Z)^2)-im / 4 * params.k * Bessels.hankelh1(1, params.k * norm(Z)) * (-Z[1] * Z[2] / norm(Z)^3);
               -im / 4 * params.k^2 *
               (Bessels.hankelh1(1, params.k * norm(Z)) / (params.k * norm(Z)) - Bessels.hankelh1(2, params.k * norm(Z))) *
               Z[2]^2 / (norm(Z)^2)-im / 4 * params.k * Bessels.hankelh1(1, params.k * norm(Z)) * (1 / norm(Z) - Z[2]^2 / norm(Z)^3)]
res_eig = QPGreen.eigfunc_expansion_hessian(Z, params; period=2π, nb_terms=1e7) - singularity


@time for i ∈ [32, 64, 128, 256, 512, 1024]
    value, grad, hess, cache = init_qp_green_fft_mod(params, i; gradient=true, hessian=true)
    res_fm = hess_smooth_qp_green_mod(Z, params, hess; nb_terms=32)
    str_err_1 = @sprintf "%.2E" abs(res_fm[1] - res_eig[1, 1])/abs(res_eig[1])
    println("Grid size: ", i, " and res: ", res_fm[1], " and error: ", str_err_1)

    str_err_2 = @sprintf "%.2E" abs(res_fm[2] - res_eig[1, 2])/abs(res_eig[2])
    println("Grid size: ", i, " and res: ", res_fm[2], " and error: ", str_err_2)

    str_err_3 = @sprintf "%.2E" abs(res_fm[3] - res_eig[2, 2])/abs(res_eig[3])
    println("Grid size: ", i, " and res: ", res_fm[3], " and error: ", str_err_3)
    println(" ")
end