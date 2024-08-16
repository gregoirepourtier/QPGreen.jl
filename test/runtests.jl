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

    alpha, c, c̃, k, ε = (0.3, 0.6, 1.0, 10.0, 0.1)
    csts = (alpha, c, c̃, k)

    χ_der(x) = GreenFunction.build_χ_der(x, c̃, c)

    Yε(x) = GreenFunction.build_Yε(x, ε)
    Yε_der(x) = GreenFunction.build_Yε_der(x, ε)
    Yε_der_2nd(x) = GreenFunction.build_Yε_der_2nd(x, ε)

    preparation_result = GreenFunction.fm_method_preparation(csts, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=32)

    calculation_result = GreenFunction.fm_method_calculation(x, csts, preparation_result, Yε; nb_terms=32)

    calculation_result
end

x = SVector(10.0, 0.4)
@time test_evaluation_GF(x)

GreenFunction.green_function_eigfct_exp(x; nb_terms=1000)
GreenFunction.green_function_img_exp(x; nb_terms=500000)


## Test build cut-off functions and derivatives
# c̃ = 2.0
# c = 1.0
# x = collect((-c̃ - 1.0):0.01:(c̃ + 1.0));
# # y = GreenFunction.build_χ.(x, c̃, c);
# # y = GreenFunction.build_χ_der.(x, c̃, c);
# χ_der(x) = GreenFunction.build_χ(x, c̃, c)
# y = χ_der.(x)

# f = Figure()
# ax = Axis(f[1, 1]; xticks=(-c̃ - 1.0):0.5:(c̃ + 1.0))
# lines!(ax, x, y)

# ε = 0.1
# x = collect(0:0.01:1.0);
# # y = GreenFunction.build_Yε.(x, ε);
# # y = GreenFunction.build_Yε_der.(x, ε);
# y = GreenFunction.build_Yε_der_2nd.(x, ε);
# f = Figure()
# ax = Axis(f[1, 1]; xticks=(0.0:0.1:1.0))
# lines!(ax, x, y)


#= 
Choice of 
    χ_der
    Yε
    Yε_der
    Yε_der_2nd
---> check the paper - build a polynomial function



Implementation FFT and IFFT from Matlab
Fourier Basis Functions

---> verify the FFT and IFFT coefficient with some dummy functions first and then apply to our problem
fft-shifted

=#