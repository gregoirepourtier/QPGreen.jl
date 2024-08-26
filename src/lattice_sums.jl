# Lattice Sums algorithm

"""
    lattice_sums_preparation(csts)

Perform the preparation step for the lattice sums algorithm.
Input arguments:

  - `csts`: tuple of constants `(β, k, d, M, L)` representing quasi-periodicty parameter, wave number
    , period, number of terms, and number of terms in the sum respectively.

Returns the lattice sum coefficients.
"""
function lattice_sums_preparation(csts)

    β, k, d, M, L = csts

    # Compute the lattice sums coefficients
    Sₗ = zeros(Complex, L + 1)
    Sₗ[1] = S₀(β, k, d, M)
    for l ∈ 1:L
        Sₗ[l + 1] = l % 2 == 0 ? S_even(l, β, k, d, M) : S_odd(l, β, k, d, M)
    end

    return Sₗ
end

"""
    lattice_sums_calculation(x, csts, Sₗ; c=0.6, nb_terms=100)

Compute the Green's function using lattice sums.
Input arguments:

  - `x`: evaluation point
  - `csts`: tuple of constants `(β, k, d, M, L)`
  - `Sₗ`: lattice sum coefficients
  - `c`: cutoff value
  - `nb_terms`: number of terms in the sum

Keyword arguments:

  - `c`: cutoff value
  - `nb_terms`: number of terms in the sum

Returns the value of the Green's function at the point `x`.
"""
function lattice_sums_calculation(x, csts, Sₗ::AbstractArray; c=0.6, nb_terms=100)

    β, k, d, M, L = csts

    if x[2] > c
        green_function_eigfct_exp(x; k=k, α=β, nb_terms=nb_terms) # to modify for general periodicity
    else
        r = √(x[1]^2 + x[2]^2)
        θ = atan(x[2], x[1])

        res_ls = Sₗ[1] * besselj0(k * r)
        for l ∈ 1:L
            res_ls += 2 * Sₗ[l + 1] * besselj(l, k * r) * cos(l * (π / 2 - θ))
        end
        return -im / 4 * (hankelh1(0, k * r) + res_ls)
    end
end

"""
    S₀(β, k, d, M)

Compute the lattice sum S₀.

Input arguments:

  - `β`: parameter β.
  - `k`: parameter k.
  - `d`: parameter d.
  - `M`: number of terms in the sum.

Returns the value of the lattice sum S₀.
"""
function S₀(β, k, d, M)

    C_euler = 0.57721566490153286060651209008240243104215933593992
    p = 2π / d
    γ₀ = k <= abs(β) ? √(β^2 - k^2) : -im * √(k^2 - β^2)

    sum_1 = 0
    for m ∈ (-M):M
        if m ≠ 0
            βₘ = β + m * p
            γₘ = k <= abs(βₘ) ? √(βₘ^2 - k^2) : -im * √(k^2 - βₘ^2)
            sum_1 += 1 / γₘ - 1 / (p * abs(m)) - (k^2 + 2 * β^2) / (2 * p^3 * abs(m)^3)
        end
    end

    -1 - 2 * im / π * (C_euler + log(k / (2 * p))) - 2 * im / (γ₀ * d) - 2 * im * (k^2 + 2 * β^2) / (p^3 * d) * zeta(3) -
    2 * im / d * sum_1
end

"""
    S_even(l, β, k, d, M)

Compute the lattice sum Sₗ for even values of `l`.
Input arguments:

  - `l`: integer value.
  - `β`: parameter β.
  - `k`: parameter k.
  - `d`: parameter d.
  - `M`: number of terms in the sum.

Returns the value of the lattice sum Sₗ for even values of `l`.
"""
function S_even(l, β, k, d, M)

    p = 2π / d
    γ₀ = k <= abs(β) ? √(β^2 - k^2) : -im * √(k^2 - β^2)
    θ₀ = abs(β) < k ? asin(β / k) : asin_ls(β / k)

    sum_1 = 0
    for m ∈ 1:M
        βₘ = β + m * p
        β₋ₘ = β - m * p
        β₋ₘ = β - m * p
        γₘ = k <= abs(βₘ) ? √(βₘ^2 - k^2) : -im * √(k^2 - βₘ^2)
        γ₋ₘ = k <= abs(β₋ₘ) ? √(β₋ₘ^2 - k^2) : -im * √(k^2 - β₋ₘ^2)
        θₘ = abs(βₘ) < k ? asin(βₘ / k) : asin_ls(βₘ / k)
        θ₋ₘ = abs(β₋ₘ) < k ? asin(β₋ₘ / k) : asin_ls(β₋ₘ / k)
        sum_1 += exp(-2 * im * l * θₘ) / (γₘ * d) + exp(2 * im * l * θ₋ₘ) / (γ₋ₘ * d) -
                 (-1)^l / (m * π) * (k / (2 * m * p))^(2 * l)
    end

    sum_2 = 0
    for m ∈ 1:l
        sum_2 += (-1)^m * 2^(2 * m) * factorial(l + m - 1) / (factorial(2 * m) * factorial(l - m)) * (p / k)^(2 * m) *
                 bernoulli(2 * m, β / p)
    end

    -2 * im * exp(-2 * im * l * θ₀) / (γ₀ * d) - 2 * im * sum_1 -
    2 * im * (-1)^l / π * (k / (2 * p))^(2 * l) * zeta(2 * l + 1) + im / (l * π) +
    im / π * sum_2
end

"""
    S_odd(l, β, k, d, M)

Compute the lattice sum Sₗ for odd values of `l`.

Input arguments:

  - `l`: integer value.
  - `β`: parameter β.
  - `k`: parameter k.
  - `d`: parameter d.
  - `M`: number of terms in the sum.

Returns the value of the lattice sum Sₗ for odd values of `l`.
"""
function S_odd(l, β, k, d, M)

    p = 2π / d
    γ₀ = k <= abs(β) ? √(β^2 - k^2) : -im * √(k^2 - β^2)
    θ₀ = abs(β) < k ? asin(β / k) : asin_ls(β / k)

    sum_1 = 0
    for m ∈ 1:M
        βₘ = β + m * p
        β₋ₘ = β - m * p
        γₘ = k <= abs(βₘ) ? √(βₘ^2 - k^2) : -im * √(k^2 - βₘ^2)
        γ₋ₘ = k <= abs(β₋ₘ) ? √(β₋ₘ^2 - k^2) : -im * √(k^2 - β₋ₘ^2)
        θₘ = abs(βₘ) < k ? asin(βₘ / k) : asin_ls(βₘ / k)
        θ₋ₘ = abs(β₋ₘ) < k ? asin(β₋ₘ / k) : asin_ls(β₋ₘ / k)
        sum_1 += exp(-im * (2 * l - 1) * θₘ) / (γₘ * d) - exp(im * (2 * l - 1) * θ₋ₘ) / (γ₋ₘ * d) +
                 im * (-1)^l * β * d * l / (m^2 * π^2) * (k / (2 * m * p))^(2 * l - 1)
    end

    sum_2 = 0
    for m ∈ 0:(l - 1)
        sum_2 += (-1)^m * 2^(2 * m) * factorial(l + m - 1) / (factorial(2 * m + 1) * factorial(l - m - 1)) * (p / k)^(2 * m + 1) *
                 bernoulli(2 * m + 1, β / p)
    end

    2 * im * exp(-im * (2 * l - 1) * θ₀) / (γ₀ * d) +
    2 * im * sum_1 +
    2 * (-1)^l * β * d * l / π^2 * (k / (2 * p))^(2 * l - 1) * zeta(2 * l + 1) -
    2 / π * sum_2
end

"""
    bernoulli(n, x)

Compute the Bernoulli polynomial of degree `n` evaluated at `x`.
Input arguments:

  - `n`: degree of the Bernoulli polynomial.
  - `x`: value at which the polynomial is evaluated.

Returns the nth Bernoulli polynomial evaluated at `x`.
"""
function bernoulli(n::Integer, x)
    res_bernoulli = zero(x)
    for k ∈ 0:n, l ∈ 0:k
        res_bernoulli += 1 / (k + 1) * (-1)^l * binomial(k, l) * (x + l)^n
    end
    res_bernoulli
end

"""
    asin_ls(x)

Compute the inverse sine function (arcsine function) for values outside of the interval [-1,1].
Input arguments:

  - `x`: real values outside the interval [-1,1].

Returns the inverse sine function (arcsine function) evaluated at `x`. For values inside the
interval [-1,1], use the standard `asin` function.
"""
asin_ls(x) = -log(√(Complex(1 - x^2)) + x * im) * im
