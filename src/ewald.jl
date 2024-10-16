# Ewald's Method to compute quasi-periodic Green's function for 2D Helmholtz equation

"""
    ewald(z, csts)

Evaluation of the Green's function using Ewald's method.
Input arguments:

  - `z`: coordinates.
  - `csts`: tuple of constants `(a, M₁, M₂, N, β, k, d)`.

Returns the value of the Green's function at the point `z`.
"""
function ewald(z, csts)

    a, M₁, M₂, N, β, k, d = csts

    p = 2π / d

    sum_1 = 0
    for m ∈ (-M₁):M₁
        βₘ = β + m * p
        γₘ = k <= abs(βₘ) ? √(βₘ^2 - k^2) : -im * √(k^2 - βₘ^2)

        sum_1 += exp(im * βₘ * z[2]) / (γₘ * d) * (exp(γₘ * z[1]) * SpecialFunctions.erfc(γₘ * d / (2 * a) + a * z[1] / d) +
                  exp(-γₘ * z[1]) * SpecialFunctions.erfc(γₘ * d / (2 * a) - a * z[1] / d))
    end

    sum_2 = 0
    for m ∈ (-M₂):M₂
        rₘ = √(z[1]^2 + (z[2] - m * d)^2)
        for n ∈ 0:N
            sum_2 += exp(im * m * β * d) * (1 / factorial(n) * (k * d / (2 * a))^(2 * n) * SpecialFunctions.expint(n + 1, a^2 * rₘ^2 / d^2))
        end
    end

    -1 / 4 * sum_1 - 1 / (4 * π) * sum_2
end
