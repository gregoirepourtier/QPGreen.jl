using Test, QPGreen, Printf

X, Y = (0.002π, 0.0)
α, c, c̃, k, ε, order = (0.3, 0.6, 1.0, 1.0, 0.4341, 8)
params = (α, c, c̃, k, order)

res_eig = eigfunc_expansion((X, Y), k, α; nb_terms=10000)
res_img = image_expansion((X, Y), k, α; nb_terms=200000)


# @code_warntype QPGreen.fm_method_preparation(params; grid_size=5);
preparation_result = QPGreen.fm_method_preparation(params; grid_size=5)
QPGreen.fm_method_calculation((X, Y), params, preparation_result; nb_terms=32)

res_eig = 0.7685069487 + 0.1952423542im
for i ∈ [32, 64, 128, 256, 512, 1024]
    preparation_result = QPGreen.fm_method_preparation(params; grid_size=i, ε=ε)
    res_fm = QPGreen.fm_method_calculation((X, Y), params, preparation_result; nb_terms=32)
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end



# using ProfileView
# function profile_test(n)
#     for i ∈ 1:n
#         QPGreen.fm_method_preparation(params; grid_size=1024)
#     end
# end

# ProfileView.@profview profile_test(1)
# ProfileView.@profview profile_test(3)
