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


# Test build cut-off functions and derivatives
c̃ = 2.0
c = 1.0
x = collect((-c̃ - 1.0):0.01:(c̃ + 1.0));
y_1 = GreenFunction.build_χ.(x, c̃, c);
y_2 = GreenFunction.build_χ_der.(x, c̃, c);


f = Figure()

ax1 = Axis(f[1, 1]; xticks=(-c̃ - 1.0):0.5:(c̃ + 1.0))
ax2 = Axis(f[1, 2]; xticks=(-c̃ - 1.0):0.5:(c̃ + 1.0))
lines!(ax1, x, y_1)
lines!(ax2, x, y_2)
f

ε = 0.1
x = collect(0:0.01:1.0);
y_1 = GreenFunction.build_Yε.(x, ε);
y_2 = GreenFunction.build_Yε_der.(x, ε);
y_3 = GreenFunction.build_Yε_der_2nd.(x, ε);

f = Figure()
ax1 = Axis(f[1, 1]; xticks=(0.0:0.1:1.0))
ax2 = Axis(f[1, 2]; xticks=(0.0:0.1:1.0))
ax3 = Axis(f[1, 3]; xticks=(0.0:0.1:1.0))

lines!(ax1, x, y_1)
lines!(ax2, x, y_2)
lines!(ax3, x, y_3)
f
