using Pkg

Pkg.activate("test/Project.toml")

using Test
using GreenFunction
using LinearAlgebra
using GLMakie
using StaticArrays


function run_all_tests()

    x = SVector(10.0, 10.0)
    y = SVector(2.0, 2.0)

    x - y

    res_img = GreenFunction.green_function_img_exp(x - y; nb_terms=10000)
    res_eig = GreenFunction.green_function_eigfct_exp(x - y; nb_terms=100)

    res1 = norm(res_img - res_eig)

    x = [0.01 * π, 0.0]
    res_img = GreenFunction.green_function_img_exp(x; nb_terms=100)
    res_eig = GreenFunction.green_function_eigfct_exp(x; nb_terms=1000)

    res2 = norm(res_img - res_eig)


    res1, res2

end

# run_all_tests()

function plot_grid()

    # Generate the meshgrid
    x, y = (1, 1)
    N = 32
    grid_X, grid_Y = GreenFunction.gen_grid_FFT(x, y, N)

    # Plot the meshgrid
    meshgrid_plot = scatter(grid_X[:], grid_Y[:])
end

# plot_grid()

function test_evaluation_GF(x)

    alpha, c, c̃, k = (0.3, 0.6, 1.0, 10.0)
    csts = (alpha, c, c̃, k)

    χ_der(x) = cos(x)

    Yε(x) = cos(x)
    Yε_der(x) = -sin(x)
    Yε_der_2nd(x) = -cos(x)

    preparation_result = GreenFunction.fm_method_preparation(csts, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=32)

    calculation_result = GreenFunction.fm_method_calculation(x, csts, preparation_result, Yε; nb_terms=32)

    calculation_result
end

x = SVector(10.0, 0.2)
@time -test_evaluation_GF(x)

GreenFunction.green_function_eigfct_exp(x; nb_terms=1000)
GreenFunction.green_function_img_exp(x; nb_terms=500000)
