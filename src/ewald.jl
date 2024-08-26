# Ewald's Method

"""
    ewald(X, Y, csts)

Evaluation of the Green's function using Ewald's method.
Input arguments:

  - `X`: x-coordinate of the evaluation point.
  - `Y`: y-coordinate of the evaluation point.
  - `csts`: tuple of constants `(a, M₁, M₂, N, β, k, d)`.

Returns the value of the Green's function at the point `(X, Y)`.
"""
function ewald(X, Y, csts)

    a, M₁, M₂, N, β, k, d = csts

    p = 2π / d

    sum_1 = 0
    for m ∈ (-M₁):M₁
        βₘ = β + m * p
        γₘ = k <= abs(βₘ) ? √(βₘ^2 - k^2) : -im * √(k^2 - βₘ^2)

        sum_1 += exp(im * βₘ * Y) / (γₘ * d) * (exp(γₘ * X) * erfc(γₘ * d / (2 * a) + a * X / d) +
                  exp(-γₘ * X) * erfc(γₘ * d / (2 * a) - a * X / d))
    end

    sum_2 = 0
    for m ∈ (-M₂):M₂
        rₘ = √(X^2 + (Y - m * d)^2)
        for n ∈ 0:N
            sum_2 += exp(im * m * β * d) * (1 / factorial(n) * (k * d / (2 * a))^(2 * n) * expint(n + 1, a^2 * rₘ^2 / d^2))
        end
    end

    -1 / 4 * sum_1 - 1 / (4 * π) * sum_2
end
