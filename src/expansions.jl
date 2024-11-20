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
