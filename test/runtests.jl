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

    res_img = GreenFunction.green_function_img_exp(x - y; nb_terms=10)
    res_eig = GreenFunction.green_function_eigfct_exp(x - y; nb_terms=10)

    res1 = norm(res_img - res_eig)

    x = [0.01 * π, 0.0]
    res_img = GreenFunction.green_function_img_exp(x; nb_terms=100)
    res_eig = GreenFunction.green_function_eigfct_exp(x; nb_terms=1000)

    res2 = norm(res_img - res_eig)


    res1, res2

end

run_all_tests()

function plot_grid()

    # Generate the meshgrid
    x, y = (1, 1)
    N = 4
    grid_X, grid_Y = GreenFunction.gen_grid_FFT(x, y, N)

    # Plot the meshgrid
    meshgrid_plot = scatter(grid_X[:], grid_Y[:])
end

plot_grid()