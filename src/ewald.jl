# Ewald's Method

"""
    ewald()
"""
function ewald()

    sum_1 = 0
    for m ∈ (-M₁):M₁
        sum_1 += exp(im * βₘ * Y) / (γₘ * d) * (exp(γₘ * X) * erfc(γₘ * d / (2 * a) + a * X / d) +
                 exp(-γₘ * X) * erfc(γₘ * d / (2 * a) - a * X / d))
    end

    sum_2 = 0
    for m ∈ (-M₂):M₂
        sum_2 += 1
    end

    -1 / 4 * sum_1 - 1 / (4 * π) * sum_2
end
