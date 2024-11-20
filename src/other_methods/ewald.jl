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

    # unpack various parameters for the Ewald's method 
    a, M₁, M₂, N, β, k, d = csts

    p = 2π / d

    # Initialize the sum
    sum_M₁ = 0.0
    sum_M₂ = 0.0

    factor_1 = d / (2 * a)
    factor_2 = a * z[1] / d
    for m ∈ (-M₁):M₁
        βₘ = β + m * p
        γₘ = k <= abs(βₘ) ? √(βₘ^2 - k^2) : -im * √(k^2 - βₘ^2)

        exp_term_1 = exp(im * βₘ * z[2]) / (γₘ * d)
        exp_term_2 = exp(γₘ * z[1])

        erfc_arg = γₘ * factor_1

        sum_M₁ += exp_term_1 * (exp_term_2 * SpecialFunctions.erfc(erfc_arg + factor_2) +
                   1 / exp_term_2 * SpecialFunctions.erfc(erfc_arg - factor_2))
    end

    factor_k = (k * d / (2 * a))^2
    factor_a = a^2 / d^2
    for m ∈ (-M₂):M₂
        dₘ = m * d
        rₘ₂ = z[1]^2 + (z[2] - dₘ)^2

        exp_term = exp(im * β * dₘ)

        for n ∈ 0:N
            sum_M₂ += exp_term * (factor_k^n / factorial(n) * SpecialFunctions.expint(n + 1, factor_a * rₘ₂))
        end
    end

    -1 / 4 * sum_M₁ - 1 / (4 * π) * sum_M₂
end
