using Pkg
Pkg.activate("test/Project.toml")
using QPGreen, GLMakie, FastGaussQuadrature, BenchmarkTools, LinearAlgebra, StaticArrays

order = 8
c, c̃ = (1.0, 2.0)

c₁, c₂ = (c, (c + c̃) / 2)

params_χ = QPGreen.IntegrationParameters(c₁, c₂, order)
cache_χ = QPGreen.IntegrationCache(params_χ)

# Test build cut-off functions and derivatives
x = collect((-c̃ - 0.5):0.01:(c̃ + 0.5));
y_1 = QPGreen.χ.(x, Ref(cache_χ));
y_2 = QPGreen.χ_der.(x, Ref(cache_χ));

f1 = Figure();

ax1 = Axis(f1[1, 1]; xticks=(-c̃ - 0.5):0.5:(c̃ + 0.5))
ax2 = Axis(f1[2, 1]; xticks=(-c̃ - 0.5):0.5:(c̃ + 0.5))
lines!(ax1, x, y_1)
lines!(ax2, x, y_2)

ε = 0.1
params_Yε = QPGreen.IntegrationParameters(ε, 2 * ε, order)
cache_Yε = QPGreen.IntegrationCache(params_Yε)

x = collect(0:0.001:1.0);
y_1 = QPGreen.Yε.(x, Ref(cache_Yε));
y_2 = QPGreen.Yε_1st_der.(x, Ref(cache_Yε));
y_3 = QPGreen.Yε_2nd_der.(x, Ref(cache_Yε));

f2 = Figure()
ax1 = Axis(f2[1, 1]; xticks=(0.0:0.1:1.0))
ax2 = Axis(f2[2, 1]; xticks=(0.0:0.1:1.0))
ax3 = Axis(f2[3, 1]; xticks=(0.0:0.1:1.0))

lines!(ax1, x, y_1)
lines!(ax2, x, y_2)
lines!(ax3, x, y_3)

display(GLMakie.Screen(), f1)
display(GLMakie.Screen(), f2)
