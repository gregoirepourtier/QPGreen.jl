using Pkg
Pkg.activate("test/Project.toml")
using QPGreen, GLMakie, FastGaussQuadrature, BenchmarkTools, LinearAlgebra, StaticArrays

order = 8
c̃ = 2.0
c = 1.0

c₁ = c
c₂ = (c + c̃) / 2
ξ, w = gausslegendre(order)

# Computation of various integral cst
poly(x, c₁, c₂, order) = (x + c₁)^order * (x + c₂)^order
int_pol_l_cst(y, c₁, c₂, order) = QPGreen.quad_CoV(x -> poly(x, c₁, c₂, order), y, -c₂, -c₁);
int_pol_r_cst(y, c₁, c₂, order) = QPGreen.quad_CoV(x -> poly(x, -c₁, -c₂, order), y, c₁, c₂);
int_pol_l(y, c₁, c₂, order) = QPGreen.quad_CoV(x -> poly(x, c₁, c₂, order), y, -c₂, -c₁);
f(x) = dot(w, QPGreen.quad_CoV.(x -> poly(x, c₁, c₂, order), ξ, -c₂, x))
h(x) = dot(w, QPGreen.quad_CoV.(x -> poly(x, -c₁, -c₂, order), ξ, c₁, x))

eval_int_cst = similar(ξ);
for i ∈ eachindex(ξ)
    eval_int_cst[i] = int_pol_l_cst(ξ[i], c₁, c₂, order)
end
eval_int_cst .= int_pol_l_cst.(ξ, c₁, c₂, order);
integral_left = dot(w, eval_int_cst);
cst_left = 1 / integral_left;

eval_int_cst = int_pol_r_cst.(ξ, c₁, c₂, order);
integral_right = dot(w, eval_int_cst);
cst_right = 1 / integral_right;

## Test build cut-off functions and derivatives ##
x = collect((-c̃ - 0.5):0.01:(c̃ + 0.5));
y_1 = QPGreen.χ.(x, c₁, c₂, cst_left, cst_right, f, h);
y_2 = QPGreen.χ_der.(x, c₁, c₂, cst_left, cst_right, x -> poly(x, c₁, c₂, order), x -> poly(x, -c₁, -c₂, order));

f1 = Figure();

ax1 = Axis(f1[1, 1]; xticks=(-c̃ - 0.5):0.5:(c̃ + 0.5))
ax2 = Axis(f1[2, 1]; xticks=(-c̃ - 0.5):0.5:(c̃ + 0.5))
lines!(ax1, x, y_1)
lines!(ax2, x, y_2)

ε = 0.1
poly_Yε(x) = poly(x, -ε, -2 * ε, order)
poly_derivative_Yε(x) = order * (x - ε)^(order - 1) * (x - 2 * ε)^order + order * (x - ε)^order * (x - 2 * ε)^(order - 1)
int_pol(x) = dot(w, QPGreen.quad_CoV.(poly_Yε, ξ, ε, x))
eval_int_cst_Yε = QPGreen.quad_CoV.(poly_Yε, ξ, ε, 2 * ε);
integral_Yε = dot(w, eval_int_cst_Yε);
cst_Yε = 1 / integral_Yε;

x = collect(0:0.001:1.0);
y_1 = QPGreen.Yε.(x, ε, cst_Yε, int_pol);
y_2 = QPGreen.Yε_1st_der.(x, ε, cst_Yε, poly_Yε);
y_3 = QPGreen.Yε_2nd_der.(x, ε, cst_Yε, poly_derivative_Yε);

f2 = Figure()
ax1 = Axis(f2[1, 1]; xticks=(0.0:0.1:1.0))
ax2 = Axis(f2[2, 1]; xticks=(0.0:0.1:1.0))
ax3 = Axis(f2[3, 1]; xticks=(0.0:0.1:1.0))

lines!(ax1, x, y_1)
lines!(ax2, x, y_2)
lines!(ax3, x, y_3)

display(GLMakie.Screen(), f1)
display(GLMakie.Screen(), f2)
