using Pkg
Pkg.activate("test/Project.toml")
using Test, QPGreen, Printf, LinearAlgebra

X, Y = (0.002π, 0.0);
α, k, c, c̃, ε, order = (0.3, 1.0, 0.6, 1.0, 0.4341, 8);
params = (α, k, c, c̃, ε, order);

res_eig = eigfunc_expansion((X, Y), k, α; nb_terms=10000);
res_img = image_expansion((X, Y), k, α; nb_terms=200000);


grid_size = 32;
# @code_warntype QPGreen.fm_method_preparation(params, grid_size);
preparation_result, interp, cache = QPGreen.fm_method_preparation(params, grid_size);
@code_warntype QPGreen.fm_method_calculation((X, Y), params, preparation_result, interp, cache; nb_terms=32);

@time QPGreen.fm_method_calculation((X, Y), params, preparation_result, interp, cache; nb_terms=32);


res_eig = 0.7685069487 + 0.1952423542im
using BenchmarkTools: @btime
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    preparation_result, interp = QPGreen.fm_method_preparation(params, i)
    res_fm = QPGreen.fm_method_calculation((X, Y), params, preparation_result, interp; nb_terms=32)
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end



using ProfileView
function profile_test(n)
    for i ∈ 1:n
        QPGreen.fm_method_preparation(params; grid_size=1024)
    end
end

ProfileView.@profview profile_test(1)
ProfileView.@profview profile_test(3)
