# using Pkg
# Pkg.activate("test/Project.toml")
using Test, QPGreen, LinearAlgebra, SpecialFunctions

@test 1 == 1


# function test_evaluation_GF(x, csts, grid_size)

#     alpha, c, c̃, k, ε = csts
#     params = (alpha, c, c̃, k)

#     χ_der(x) = QPGreen.build_χ_der(x, c̃, c)

#     Yε(x) = QPGreen.build_Yε(x, ε)
#     Yε_der(x) = QPGreen.build_Yε_der(x, ε)
#     Yε_der_2nd(x) = QPGreen.build_Yε_der_2nd(x, ε)

#     preparation_result = QPGreen.fm_method_preparation(params, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=grid_size)
#     calculation_result = QPGreen.fm_method_calculation(x, params, preparation_result, Yε; α=alpha, k=k, nb_terms=32)

#     calculation_result
# end

# X, Y = (0.01π, 0.01)
# α, c, c̃, k, ε = (0.3, 0.6, 1.0, √10, 10)
# csts = (α, c, c̃, k, ε)

# @time test_evaluation_GF((X, Y), csts, 50)

# res_eig = QPGreen.green_function_eigfct_exp((X, Y); k=k, α=α, nb_terms=1000)
# res_img = QPGreen.green_function_img_exp((X, Y); k=k, α=α, nb_terms=200000)
