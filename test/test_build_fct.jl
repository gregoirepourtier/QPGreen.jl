using Pkg
Pkg.activate("test/Project.toml")
using QPGreen, GLMakie

## Test build cut-off functions and derivatives ##
c̃ = 2.0
c = 1.0
x = collect((-c̃ - 0.5):0.01:(c̃ + 0.5));
y_1 = QPGreen.build_χ.(x, c̃, c);
y_2 = QPGreen.build_χ_der.(x, c̃, c);

f1 = Figure();

ax1 = Axis(f1[1, 1]; xticks=(-c̃ - 0.5):0.5:(c̃ + 0.5))
ax2 = Axis(f1[2, 1]; xticks=(-c̃ - 0.5):0.5:(c̃ + 0.5))
lines!(ax1, x, y_1)
lines!(ax2, x, y_2)

ε = 0.1
x = collect(0:0.001:1.0);
y_1 = QPGreen.build_Yε.(x, ε);
y_2 = QPGreen.build_Yε_der.(x, ε);
y_3 = QPGreen.build_Yε_der_2nd.(x, ε);

f2 = Figure()
ax1 = Axis(f2[1, 1]; xticks=(0.0:0.1:1.0))
ax2 = Axis(f2[2, 1]; xticks=(0.0:0.1:1.0))
ax3 = Axis(f2[3, 1]; xticks=(0.0:0.1:1.0))

lines!(ax1, x, y_1)
lines!(ax2, x, y_2)
lines!(ax3, x, y_3)

display(GLMakie.Screen(), f1)
display(GLMakie.Screen(), f2)

# using QPGreen
# using Test

# @testset "QPGreen.jl" begin
#     # Write your tests here.
# end
