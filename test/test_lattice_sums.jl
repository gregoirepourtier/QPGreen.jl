using Pkg

Pkg.activate("test/Project.toml")

using Test
using StaticArrays
using GreenFunction
using SpecialFunctions

X = 0.0
Y = 0.01 * 2π
r = √(X^2 + Y^2)
θ = atan(Y, X)

β, k, d, M, L = (√2 / (2 * π), 2 / (2 * π), 2 * π, 7, 3)
csts = (β, k, d, M, L)

(X / d == 0.0, Y / d == 0.01, k * d == 2, β * d == √2, r < d)


GreenFunction.lattice_sums_preparation(r, θ, csts)


function test_img_exp(z, β, d; nb_terms=100)

    G = zero(Complex{eltype(z)})

    range_term = nb_terms ÷ 2

    # Compute the value of the Green function by basic image expansion
    for n ∈ (-range_term):range_term
        rₙ = √(z[1]^2 + (z[2] - n * d)^2)
        G += -im / 4 * exp(im * n * β * d) * hankelh1(0, k * rₙ)
    end

    G
end

test_img_exp((X, Y), β, d; nb_terms=500000)

function test_eigfct_exp(z, β, d; nb_terms=100)

    G = zero(Complex{eltype(z)})

    p = 2π / d
    range_term = nb_terms ÷ 2

    # Compute the value of the Green function by basic eigenfunction expansion
    for m ∈ (-range_term):range_term
        βₘ = β + m * p
        γₘ = k <= abs(βₘ) ? √(βₘ^2 - k^2) : -im * √(k^2 - βₘ^2)

        G += -1 / (2 * d) * (1 / γₘ) * exp(-γₘ * abs(z[1]) + im * βₘ * z[2])
    end

    G
end

test_eigfct_exp((X, Y), β, d; nb_terms=1000)

x = -1.2
GreenFunction.asin_ls(x)
asin(Complex(x))
