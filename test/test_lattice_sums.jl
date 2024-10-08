using Pkg

Pkg.activate("test/Project.toml")

using Test, QPGreen, SpecialFunctions

# X, Y = (0.0, 0.01 * 2π)
X, Y = (0.0, 0.002)
r, θ = (√(X^2 + Y^2), atan(Y, X))

# β, k, d, M, L = (√2 / (2 * π), 2 / (2 * π), 2 * π, 80, 4)
β, k, d, M, L = (0.3, 1, 2 * π, 80, 4)
csts = (β, k, d, M, L)

# Test to match parameter from the paper (Linton, 1998)
(X / d == 0.0, Y / d == 0.01, k * d == 2, β * d == √2, r < d)

Sl = QPGreen.lattice_sums_preparation(csts);
eval_ls = -QPGreen.lattice_sums_calculation((X, Y), csts, Sl; nb_terms=100)

@test eval_ls == (-0.4595073481223693 - 0.35087920819684576 * im)

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

-test_img_exp((X, Y), β, d; nb_terms=500000)

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
