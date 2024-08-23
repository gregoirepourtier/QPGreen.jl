# Lattice Sums algorithm

# get_ϕₘ(m) = m<0 ? π : 1

"""
    lattice_sums_preparation()
"""
function lattice_sums_preparation(r; k=10, α=0.3, nb_terms=100)

    # Compute the lattice sums coefficients
    # lattice_sum = 0
    lattice_sum = S_0() * besselj0(kr)

    for l ∈ 1:L
        lattice_sum += 2 * Sₗ * besselj(l, kr) * cos(l * (π / 2 - θ))
    end

    return -im / 4 * (hankelh1(0, k * r) + lattice_sum)
end

function lattice_sums_calculation()
end

"""
    S_0()
"""
function S_0(β, k, d, θ₀, M)

    C_euler = 0.5772157
    p = 2 * pi / d
    γ₀ = √(β^2 - k^2) # = im * √(k^2 - β^2) ?

    sum_1 = 0
    for m ∈ (-M):M
        βₘ = β + m * p
        γₘ = √(βₘ^2 - k^2) # = im * √(k^2 - βₘ^2) ?
        sum_1 += 1 / γₘ - 1 / (p * abs(m)) - (k^2 + 2 * β^2) / (2 * p^3 * abs(m)^3) # Typo paper? abs(m)^2 or abs(m)^3?
    end

    -1 - 2 * im / π * (C_euler + log(k / (2 * p))) - 2 * im / (γ₀ * d) - 2 * im * (k^2 + 2 * β^2) / (p^3 * d) * zeta(3) -
    2 * im / d * sum_1
end

"""
    S_even()
"""
function S_even()

    sum_1 = 0
    for m ∈ 1:M
        sum_1 += exp(-2 * im * l * θₘ) / (γₘ * d) + exp(-2 * im * l * θ₋ₘ) / (γ₋ₘ * d) -
                 (-1)^l / (m * π) * (k / (2 * m * p))^(2 * l)
    end

    sum_2 = 0
    for m ∈ 1:l
        sum_2 += (-1)^m * 2^(2 * m) * factorial(l + m - 1) / (factorial(2 * m) * factorial(l - m)) * (p / k)^(2 * m) *
                 bernouilli_2m(β / p)
    end

    -2 * im * exp(-2 * im * l * θ₀) / (γ₀ * d) - 2 * im * sum_1 - 2 * im * (-1)^l / π * (k / (2 * p))^(2 * l) * zeta(2 * l + 1) +
    im / (l * π) + i / π * sum_2
end

function S_odd()

    sum_1 = 0
    for m ∈ 1:M
        sum_1 += exp(-im * (2 * l - 1) * θₘ) / (γₘ * d) - exp(im * (2 * l - 1) * θ₋ₘ) / (γ₋ₘ * d) +
                 im * (-1)^l * β * d * l / (m^2 * π^2) * (k / (2 * m * p))^(2 * l - 1)
    end

    sum_2 = 0
    for m ∈ 1:(l - 1)
        sum_2 += (-1)^m * 2^(2 * m) * factorial(l + m - 1) / (factorial(2 * m + 1) * factorial(l - m - 1)) * (p / k)^(2 * m + 1) *
                 bernouilli_2mP1(β / p)
    end

    2 * im * exp(-im * (2 * l - 1) * θ₀) / (γ₀ * d) +
    2 * im * sum_1 +
    2 * (-1)^l * β * d * l / π^2 * (k / (2 * p))^(2 * l - 1) * zeta(2 * l + 1) -
    2 / π * sum_2
end
