using Pkg

Pkg.activate("test/Project.toml")

using Test
using StaticArrays
using GreenFunction

# P = SVector(0.0, 0.1)
# Q = SVector(0.0, 0.2)
X = 0.0 # Q[1] - P[1]
Y = 0.01 * 2 * π # Q[2] - P[2]
r = √(X^2 + Y^2)
θ = atan(Y / X)

β, k, d, M, L = (√2 / (2 * π), 2 / (2 * π), 2 * π, 7, 3)
csts = (β, k, d, M, L)

X / d == 0.0
Y / d == 0.01
k * d == 2
β * d == √2
r < d

GreenFunction.lattice_sums_preparation(r, θ, csts)

GreenFunction.green_function_eigfct_exp((X, Y); nb_terms=1000)
GreenFunction.green_function_img_exp((X, Y); nb_terms=500000)
