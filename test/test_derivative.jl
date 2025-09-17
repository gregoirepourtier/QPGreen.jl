using Pkg
Pkg.activate("test/Project.toml")
using Test
using QPGreen, Bessels
using Printf, LinearAlgebra
using BenchmarkTools: @btime

Z = (0.02π, 0.01);

params = (alpha=0.3, k=(√(10)), c=0.6, c_tilde=1.0, epsilon=0.4341, order=8);

res_eig = eigfunc_expansion_gradient(Z, params; nb_terms=1e6)

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
value, grad, cache = init_qp_green_fft_mod(params, grid_size; gradient=true);
grad_G = grad_qp_green_mod(Z, params, grad, cache; nb_terms=32)

@time for i ∈ [32, 64, 128, 256, 512, 1024]
    value, grad, cache = init_qp_green_fft_mod(params, i; gradient=true)
    res_fm = grad_qp_green_mod(Z, params, grad, cache; nb_terms=32)
    str_err_1 = @sprintf "%.2E" abs(res_fm[1] - res_eig[1])/abs(res_eig[1])
    println("Grid size: ", i, " and res: ", res_fm[1], " and error: ", str_err_1)

    str_err_2 = @sprintf "%.2E" abs(res_fm[2] - res_eig[2])/abs(res_eig[2])
    println("Grid size: ", i, " and res: ", res_fm[2], " and error: ", str_err_2)
    println(" ")
end

## smooth part
# Z = (0.002π, 1);
# params = (alpha=0.6, k=2π, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8);
res_img_smooth = image_expansion_gradient_smooth(Z, params; nb_terms=1e7)
res_eig_smooth = eigfunc_expansion_gradient(Z, params; nb_terms=1e7) .+
                 im / 4 * params.k * Bessels.hankelh1(1, params.k * norm(Z)) / norm(Z) .* Z

grad_G = grad_smooth_qp_green(Z, params, grad, cache; nb_terms=32);



res_test = res_eig_smooth
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    value, grad, cache = init_qp_green_fft(params, i; derivative=true)
    res_fm = grad_smooth_qp_green(Z, params, grad, cache; nb_terms=32)
    str_err_1 = @sprintf "%.2E" abs(res_fm[1] - res_test[1])/abs(res_test[1])
    println("Grid size: ", i, " and res: ", res_fm[1], " and error: ", str_err_1)

    str_err_2 = @sprintf "%.2E" abs(res_fm[2] - res_test[2])/abs(res_test[2])
    println("Grid size: ", i, " and res: ", res_fm[2], " and error: ", str_err_2)
    println(" ")
end


@time for i ∈ [32, 64, 128, 256, 512, 1024]
    value, grad, cache = init_qp_green_fft_mod(params, i; gradient=true)
    res_fm = grad_smooth_qp_green_mod(Z, params, grad, cache; nb_terms=32)
    str_err_1 = @sprintf "%.2E" abs(res_fm[1] - res_test[1])/abs(res_test[1])
    println("Grid size: ", i, " and res: ", res_fm[1], " and error: ", str_err_1)

    str_err_2 = @sprintf "%.2E" abs(res_fm[2] - res_test[2])/abs(res_test[2])
    println("Grid size: ", i, " and res: ", res_fm[2], " and error: ", str_err_2)
    println(" ")
end