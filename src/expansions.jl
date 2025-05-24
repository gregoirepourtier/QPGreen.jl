# Formulas for the α-quasi-periodic Green's function of the 2D Helmholtz equation (basic image and eigenfunction expansions)

"""
    image_expansion(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the α-quasi-periodic Green's function using the image expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the α-quasi-periodic Green's function for the 2D Helmholtz equation at the point z,
    computed via the basic image expansion.
"""
function image_expansion(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the Green function by basic image expansion
    G = im / 4 * Bessels.hankelh1(0, k * √(z[1]^2 + z[2]^2))
    for n ∈ 1:nb_terms
        r₋ₙ = √((z[1] - period * -n)^2 + z[2]^2)
        rₙ = √((z[1] - period * n)^2 + z[2]^2)
        G += im / 4 * exp(im * period * α * -n) * Bessels.hankelh1(0, k * r₋ₙ) +
             im / 4 * exp(im * period * α * n) * Bessels.hankelh1(0, k * rₙ)
    end

    G
end

"""
    image_expansion_derivative(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the derivative of the α-quasi-periodic Green's function using the image expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the derivative of the α-quasi-periodic Green's function for the 2D Helmholtz equation at
    the point z, computed via the basic image expansion.
"""
function image_expansion_derivative(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the derivative of the Green function by basic image expansion
    r₀ = √(z[1]^2 + z[2]^2)
    G_prime_x1 = -im / 4 * k * Bessels.hankelh1(1, k * r₀) * z[1] / r₀
    G_prime_x2 = -im / 4 * k * Bessels.hankelh1(1, k * r₀) * z[2] / r₀
    for n ∈ 1:nb_terms
        r₋ₙ = √((z[1] - period * -n)^2 + z[2]^2)
        rₙ = √((z[1] - period * n)^2 + z[2]^2)
        G_prime_x1 += -im / 4 * k * exp(im * period * α * -n) * Bessels.hankelh1(1, k * r₋ₙ) * (z[1] - period * -n) / r₋ₙ -
                      im / 4 * k * exp(im * period * α * n) * Bessels.hankelh1(1, k * rₙ) * (z[1] - period * n) / rₙ
        G_prime_x2 += -im / 4 * k * exp(im * period * α * -n) * Bessels.hankelh1(1, k * r₋ₙ) * z[2] / r₋ₙ -
                      im / 4 * k * exp(im * period * α * n) * Bessels.hankelh1(1, k * rₙ) * z[2] / rₙ
    end

    SVector(G_prime_x1, G_prime_x2)
end

"""
    image_expansion_smooth(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the smooth α-quasi-periodic Green's function (i.e. without the term ``H_0^{(1)(k|x|)}``
using the image expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the smooth α-quasi-periodic Green's function for the 2D Helmholtz equation at the point z,
    computed via the basic image expansion.
"""
function image_expansion_smooth(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the Green function by basic image expansion
    G = zero(Complex{eltype(z)})
    for n ∈ 1:nb_terms
        r₋ₙ = √((z[1] - period * -n)^2 + z[2]^2)
        rₙ = √((z[1] - period * n)^2 + z[2]^2)
        G += im / 4 * exp(im * period * α * -n) * Bessels.hankelh1(0, k * r₋ₙ) +
             im / 4 * exp(im * period * α * n) * Bessels.hankelh1(0, k * rₙ)
    end

    G
end

"""
    image_expansion_derivative_smooth(z, params:NamedTuple; period=2π, nb_terms=50)

Compute the derivative of the smooth α-quasi-periodic Green's function using the image expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the derivative of the smooth α-quasi-periodic Green's function for 2D Helmholtz equation
    at the point z, computed via the basic image expansion.
"""
function image_expansion_derivative_smooth(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the derivative of the Green function by basic image expansion
    G_prime_x1 = zero(Complex{eltype(z)})
    G_prime_x2 = zero(Complex{eltype(z)})
    for n ∈ 1:nb_terms
        r₋ₙ = √((z[1] - period * -n)^2 + z[2]^2)
        rₙ = √((z[1] - period * n)^2 + z[2]^2)
        G_prime_x1 += -im / 4 * k * exp(im * period * α * -n) * Bessels.hankelh1(1, k * r₋ₙ) * (z[1] - period * -n) / r₋ₙ -
                      im / 4 * k * exp(im * period * α * n) * Bessels.hankelh1(1, k * rₙ) * (z[1] - period * n) / rₙ
        G_prime_x2 += -im / 4 * k * exp(im * period * α * -n) * Bessels.hankelh1(1, k * r₋ₙ) * z[2] / r₋ₙ -
                      im / 4 * k * exp(im * period * α * n) * Bessels.hankelh1(1, k * rₙ) * z[2] / rₙ
    end

    SVector(G_prime_x1, G_prime_x2)
end



"""
    eigfunc_expansion(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the α-quasi-periodic Green's function using the eigenfunction expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the α-quasi-periodic Green's function for 2D Helmholtz equation at the point z,
    computed by the basic eigenfunction expansion.
"""
function eigfunc_expansion(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the Green function by basic eigenfunction expansion
    β₀ = abs(α) <= k ? √(k^2 - α^2) : im * √(α^2 - k^2)
    G = im / (2 * period) * (1 / β₀) * exp(im * α * z[1] + im * β₀ * abs(z[2]))
    for n ∈ 1:nb_terms
        α₋ₙ = α - n
        αₙ = α + n

        β₋ₙ = abs(α₋ₙ) <= k ? √(k^2 - α₋ₙ^2) : im * √(α₋ₙ^2 - k^2)
        βₙ = abs(αₙ) <= k ? √(k^2 - αₙ^2) : im * √(αₙ^2 - k^2)

        G += im / (2 * period) * (1 / β₋ₙ) * exp(im * α₋ₙ * z[1] + im * β₋ₙ * abs(z[2])) +
             im / (2 * period) * (1 / βₙ) * exp(im * αₙ * z[1] + im * βₙ * abs(z[2]))
    end

    G
end

"""
    eigfunc_expansion_derivative(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the derivative of the α-quasi-periodic Green's function using the eigenfunction expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the derivative of the α-quasi-periodic Green's function for 2D Helmholtz equation at
    the point z, computed via the basic eigenfunction expansion.
"""
function eigfunc_expansion_derivative(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the derivative of the Green function by basic eigenfunction expansion
    β₀ = abs(α) <= k ? √(k^2 - α^2) : im * √(α^2 - k^2)
    G_prime_x1 = im / (2 * period) * im * α / β₀ * exp(im * α * z[1] + im * β₀ * abs(z[2]))
    G_prime_x2 = im / (2 * period) * im * sign(z[2]) * exp(im * α * z[1] + im * β₀ * abs(z[2]))
    for n ∈ 1:nb_terms
        α₋ₙ = α - n
        αₙ = α + n

        β₋ₙ = abs(α₋ₙ) <= k ? √(k^2 - α₋ₙ^2) : im * √(α₋ₙ^2 - k^2)
        βₙ = abs(αₙ) <= k ? √(k^2 - αₙ^2) : im * √(αₙ^2 - k^2)

        exp_term_minus = exp(im * α₋ₙ * z[1] + im * β₋ₙ * abs(z[2]))
        exp_term_plus = exp(im * αₙ * z[1] + im * βₙ * abs(z[2]))

        G_prime_x1 += im / (2 * period) * im * α₋ₙ / β₋ₙ * exp_term_minus +
                      im / (2 * period) * im * αₙ / βₙ * exp_term_plus
        G_prime_x2 += im / (2 * period) * im * sign(z[2]) * (exp_term_minus + exp_term_plus)
    end

    SVector(G_prime_x1, G_prime_x2)
end
