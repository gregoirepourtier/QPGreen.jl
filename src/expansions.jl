#%% Formulas for α-quasi-periodic Green function for the 2D Helmholtz equation (basic image and eigenfunction expansions)

"""
    image_expansion(z, csts; period=2π, nb_terms=100)

  - z: coordinates of the difference between the target point and source point
  - csts: Named tuple of the constants for the problem definition

Returns the value of the α-quasi-periodic Green function for 2D Helmholtz equation at the point z
defined by the basic image expansion.
"""
function image_expansion(z, csts::NamedTuple; period=2π, nb_terms=100)

    G = zero(Complex{eltype(z)})

    α, k = (csts.α, csts.k)

    range_term = nb_terms ÷ 2

    # Compute the value of the Green function by basic image expansion
    for n ∈ (-range_term):range_term
        rₙ = √((z[1] - period * n)^2 + z[2]^2)
        G += im / 4 * exp(im * period * α * n) * Bessels.hankelh1(0, k * rₙ)
    end

    G
end

"""
    eigfunc_expansion(z, csts; period=2π, nb_terms=100)

  - z: coordinates of the difference between the target point and source point
  - csts: Named tuple of the constants for the problem definition

Returns the value of the α-quasi-periodic Green function for 2D Helmholtz equation at the point z
defined by the basic eigenfunction expansion.
"""
function eigfunc_expansion(z, csts::NamedTuple; period=2π, nb_terms=100)

    G = zero(Complex{eltype(z)})

    range_term = nb_terms ÷ 2

    α, k = (csts.α, csts.k)

    # Compute the value of the Green function by basic eigenfunction expansion
    for n ∈ (-range_term):range_term
        αₙ = α + n
        βₙ = abs(αₙ) <= k ? √(k^2 - αₙ^2) : im * √(αₙ^2 - k^2)
        G += im / (2 * period) * (1 / βₙ) * exp(im * αₙ * z[1] + im * βₙ * abs(z[2]))
    end

    G
end

"""
    eigfunc_expansion_derivative(z, csts; period=2π, nb_terms=100)

Calculate the derivative of the Green's function using the eigenfunction expansion.
Input arguments:

  - z: 2D point at which the derivative is evaluated
  - csts: Named tuple of the constants for the problem definition

Keyword arguments:

  - period: period of the Green's function
  - nb_terms: number of terms in the series expansion

Returns the value of the derivative of the Green's function.
"""
function eigfunc_expansion_derivative(z, csts::NamedTuple; period=2π, nb_terms=100)

    G_prime_x1 = zero(Complex{eltype(z)})
    G_prime_x2 = zero(Complex{eltype(z)})

    range_term = nb_terms ÷ 2

    α, k = (csts.α, csts.k)

    # Compute the value of the derivative of the Green function by basic eigenfunction expansion
    for n ∈ (-range_term):range_term
        αₙ = α + n
        βₙ = abs(αₙ) <= k ? √(k^2 - αₙ^2) : im * √(αₙ^2 - k^2)

        exp_term = exp(im * αₙ * z[1] + im * βₙ * abs(z[2]))
        G_prime_x1 += im / (2 * period) * im * αₙ / βₙ * exp_term
        G_prime_x2 += im / (2 * period) * im * sign(z[2]) * exp_term
    end

    return G_prime_x1, G_prime_x2
end
