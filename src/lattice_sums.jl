# Lattice Sums algorithm

"""
    lattice_sums_preparation()
"""
function lattice_sums_preparation(r; k=10, α=0.3, nb_terms=100)

    # Compute the lattice sums coefficients
    -im / 4 * (hankelh1(0, k * r))
end

function S_0()
    
end

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


end
