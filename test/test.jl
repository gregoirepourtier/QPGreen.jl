using Pkg
Pkg.activate("test/Project.toml")
using Test
using QPGreen
using Printf, LinearAlgebra
using BenchmarkTools: @btime
using SpecialFunctions

Z = (0.002π, 0.0);
params = (alpha=0.3, k=1.0, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8);

grid_size = 32;
val_interp, cache = init_qp_green_fft(params, grid_size);
eval_qp_green(Z, params, val_interp, cache; nb_terms=32);

res_eig = 0.7685069487 + 0.1952423542im - im / 4 * besselh(0, params.k * norm(Z))
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    val_interp, cache = init_qp_green_fft(params, i)
    res_fm = eval_smooth_qp_green(Z, params, val_interp, cache; nb_terms=32)
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end

### Modified part
grid_size = 32;
val_interp, cache = init_qp_green_fft_mod(params, grid_size);
eval_qp_green_mod(Z, params, val_interp, cache; nb_terms=32);

res_eig = 0.7685069487 + 0.1952423542im - im / 4 * besselh(0, params.k * norm(Z))
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    val_interp, cache = init_qp_green_fft_mod(params, i)
    res_fm = eval_smooth_qp_green_mod(Z, params, val_interp; nb_terms=32) # - im / 4 * besselh(0, params.k * norm(Z))
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end

@btime eval_qp_green($Z, $params, $val_interp, $cache; nb_terms=32);
@btime eval_qp_green_mod($Z, $params, $val_interp, $cache; nb_terms=32);
@btime eval_smooth_qp_green($Z, $params, $val_interp, $cache; nb_terms=32);
@btime eval_smooth_qp_green_mod($Z, $params, $val_interp; nb_terms=32);

## Smooth part
grid_size = 1024;
val_interp, cache = init_qp_green_fft_mod(params, grid_size);
res_fm = eval_smooth_qp_green_mod(Z, params, val_interp; nb_terms=500);
println(res_fm)


res_im = 0.7685069487 + 0.1952423542im - im / 4 * besselh(0, params.k * norm(Z))
str_err = @sprintf "%.2E" abs(res_fm - res_im)/abs(res_im);
println("Grid size: ", grid_size, " and res: ", res_im, " and error: ", str_err)


grid_size = 2048;
Z0 = (0.0, 0.0);
res_im = -0.057162619840346006 - 0.055147615421561616im
res_im = -0.05715877239573722 - 0.05514376771141578im
res_im = -0.05716188285483287 - 0.05514687828014918im
image_expansion_smooth(Z0, params; nb_terms=5e8)
val_interp, cache = init_qp_green_fft_mod(params, grid_size);
res_fm = eval_smooth_qp_green_mod(Z0, params, val_interp; nb_terms=500)
str_err = @sprintf "%.2E" abs(res_fm - res_im)/abs(res_im);
println("Grid size: ", grid_size, " and res: ", res_im, " and error: ", str_err)
