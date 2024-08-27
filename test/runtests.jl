using Pkg

Pkg.activate("test/Project.toml")

using Test, GreenFunction, LinearAlgebra, GLMakie, SpecialFunctions

function plot_grid()

    # Generate the meshgrid
    x, y = (π, 1.0)
    N = 32
    grid_X, grid_Y = GreenFunction.gen_grid_FFT(x, y, N)

    # Plot the meshgrid
    meshgrid_plot = scatter(grid_X[:], grid_Y[:])
end
# plot_grid()

function test_evaluation_GF(x, csts, grid_size)

    alpha, c, c̃, k, ε = csts
    params = (alpha, c, c̃, k)

    χ_der(x) = GreenFunction.build_χ_der(x, c̃, c)

    Yε(x) = GreenFunction.build_Yε(x, ε)
    Yε_der(x) = GreenFunction.build_Yε_der(x, ε)
    Yε_der_2nd(x) = GreenFunction.build_Yε_der_2nd(x, ε)

    preparation_result = GreenFunction.fm_method_preparation(params, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=grid_size)
    calculation_result = GreenFunction.fm_method_calculation(x, params, preparation_result, Yε; α=alpha, k=k, nb_terms=32)

    calculation_result
end

X, Y = (0.0, 0.01 * 2π)
r, θ = (√(X^2 + Y^2), atan(Y, X))


α, c, c̃, k, ε = (√2 / (2 * π), 0.6, 1.0, 2 / (2 * π), 0.1)
csts = (α, c, c̃, k, ε)

# Test to match parameter from the paper (Linton, 1998)
d = 2π;
β = α;
(X / d == 0.0, Y / d == 0.01, k * d == 2, β * d == √2, r < d)

for i ∈ 1:7
    N = 2^i
    res = test_evaluation_GF((X, Y), csts, N)
    println(res)
end
@time test_evaluation_GF((X, Y), csts, 50)

res_eig = GreenFunction.green_function_eigfct_exp((X, Y); k=k, α=α, nb_terms=1000)
res_img = GreenFunction.green_function_img_exp((X, Y); k=k, α=α, nb_terms=200000)



## Test build cut-off functions and derivatives ##
c̃ = 2.0
c = 1.0
x = collect((-c̃ - 0.5):0.01:(c̃ + 0.5));
y_1 = GreenFunction.build_χ.(x, c̃, c);
y_2 = GreenFunction.build_χ_der.(x, c̃, c);


f1 = Figure();

ax1 = Axis(f1[1, 1]; xticks=(-c̃ - 0.5):0.5:(c̃ + 0.5))
ax2 = Axis(f1[2, 1]; xticks=(-c̃ - 0.5):0.5:(c̃ + 0.5))
lines!(ax1, x, y_1)
lines!(ax2, x, y_2)

ε = 0.1
x = collect(0:0.001:1.0);
y_1 = GreenFunction.build_Yε.(x, ε);
y_2 = GreenFunction.build_Yε_der.(x, ε);
y_3 = GreenFunction.build_Yε_der_2nd.(x, ε);

f2 = Figure();
ax1 = Axis(f2[1, 1]; xticks=(0.0:0.1:1.0))
ax2 = Axis(f2[2, 1]; xticks=(0.0:0.1:1.0))
ax3 = Axis(f2[3, 1]; xticks=(0.0:0.1:1.0))

lines!(ax1, x, y_1)
lines!(ax2, x, y_2)
lines!(ax3, x, y_3)

display(GLMakie.Screen(), f1)
display(GLMakie.Screen(), f2)

# using GreenFunction
# using Test

# @testset "GreenFunction.jl" begin
#     # Write your tests here.
# end
