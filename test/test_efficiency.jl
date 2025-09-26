using Pkg
Pkg.activate("test/.")

using QPGreen
using StaticArrays, LinearAlgebra
using BenchmarkTools: @btime
# using Cthulhu

grid_size = 1024;
params = (alpha=0.3, k=√10, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8);
val, grad, hess, cache = init_qp_green_fft(params, grid_size; grad=true, hess=true);
val_asympt, grad_asympt, cache_asympt = QPGreen.init_qp_green_fft_asymptotic(params, grid_size; grad=true);

Z = (0.0, 0.01 * π)
# Z = (0.2 * 2π, 0.09 * 2π);
# Z = (0.0, 0.09 * 2π);
eigfunc_expansion(Z, params)
QPGreen.image_expansion_smooth(Z, params; nb_terms=1e5)

norm(Z) > params.epsilon
norm(Z) < 2 * params.epsilon
Z[2] < params.c

# eval_qp_green(Z, params, val, cache; nb_terms=5) # Yε(...) = 0.9574729914264387

@btime eval_qp_green($Z, $params, $val, $cache; nb_terms=5)
@btime QPGreen.eval_qp_green_asymptotic($Z, $params, $val_asympt, $cache_asympt; nb_terms=5)

@btime eval_smooth_qp_green($Z, $params, $val, $cache; nb_terms=5)
@btime QPGreen.eval_smooth_qp_green_asymptotic($Z, $params, $val_asympt, $cache_asympt; nb_terms=5)


Z[2] > params.c
Z = (0.2 * 2π, 0.5 * 2π);
@btime eval_qp_green($Z, $params, $val, $cache; nb_terms=5)
@btime QPGreen.eval_qp_green_asymptotic($Z, $params, $val_asympt, $cache_asympt; nb_terms=5)

@btime eval_smooth_qp_green($Z, $params, $val, $cache; nb_terms=5)
@btime QPGreen.eval_smooth_qp_green_asymptotic($Z, $params, $val_asympt, $cache_asympt; nb_terms=5)


### Gradient
# Z = (0.0, 0.01 * π)
# Z = (0.2 * 2π, 0.09 * 2π);
Z = (0.1, 0.09 * 2π);
norm(Z) > params.epsilon
norm(Z) < 2 * params.epsilon
Z[2] < params.c
eigfunc_expansion_grad(Z, params; nb_terms=1e5)
QPGreen.image_expansion_grad_smooth(Z, params; nb_terms=1e5)

@btime grad_qp_green($Z, $params, $grad, $cache; nb_terms=5)
@btime QPGreen.grad_qp_green_asymptotic($Z, $params, $grad_asympt, $cache_asympt; nb_terms=5)

@btime grad_smooth_qp_green($Z, $params, $grad, $cache; nb_terms=5)
@btime QPGreen.grad_smooth_qp_green_asymptotic($Z, $params, $grad_asympt, $cache_asympt; nb_terms=5)

#### Hessian
# Z = (0.0, 0.01 * π)
# Z = (0.2 * 2π, 0.09 * 2π);
Z = (0.0, 0.09 * 2π);
norm(Z) > params.epsilon
norm(Z) < 2 * params.epsilon
Z[2] < params.c
eigfunc_expansion_hess(Z, params; nb_terms=1e7)
QPGreen.image_expansion_hess(Z, params; nb_terms=1e7)
QPGreen.image_expansion_hess_smooth(Z, params; nb_terms=1e5)

@btime hess_qp_green($Z, $params, $hess, $cache; nb_terms=5)
# @btime QPGreen.hess_qp_green_asymptotic($Z, $params, $_asympt, $cache_asympt; nb_terms=5)

@btime hess_smooth_qp_green($Z, $params, $hess, $cache; nb_terms=5)
# @btime QPGreen.grad_smooth_qp_green_asymptotic($Z, $params, $grad_asympt, $cache_asympt; nb_terms=5)





Z = (0.1, 0.09 * 2π);
# params = (alpha=0.3, k=(√(10)), c=0.6, c_tilde=1.0, epsilon=0.4341, order=8);

res_eig = eigfunc_expansion_grad(Z, params; nb_terms=1e6)

# grid_size = 1024;
# value, grad, cache = init_qp_green_fft(params, grid_size; grad=true);
# grad_G = grad_qp_green(Z, params, grad, cache; nb_terms=32);

value, grad, cache = init_qp_green_fft(params, 1024; grad=true);
res_fm = grad_qp_green(Z, params, grad, cache; nb_terms=32)


using Printf
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    value, grad, cache = init_qp_green_fft(params, i; grad=true)
    res_fm = grad_qp_green(Z, params, grad, cache; nb_terms=32)
    str_err_1 = @sprintf "%.2E" abs(res_fm[1] - res_eig[1])/abs(res_eig[1])
    println("Grid size: ", i, " and res: ", res_fm[1], " and error: ", str_err_1)

    str_err_2 = @sprintf "%.2E" abs(res_fm[2] - res_eig[2])/abs(res_eig[2])
    println("Grid size: ", i, " and res: ", res_fm[2], " and error: ", str_err_2)
    println(" ")
end