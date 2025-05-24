# Construct the cutoff functions χ and Yε, along with their derivatives, as defined in [1].

"""
    χ(x, cache::IntegrationCache)

Evaluate the cutoff function `χ` at the point `x` (`C^∞` function).

# Input arguments

  - `x`: point at which the cutoff function is evaluated.
  - `cache`: see [`IntegrationCache`](@ref).

# Returns

  - The value of the cutoff function `χ` at `x`.
"""
function χ(x::T, cache::IntegrationCache) where {T}
    if abs(x) >= cache.params.b
        return zero(T)
    elseif abs(x) <= cache.params.a
        return one(T)
    elseif -cache.params.b < x < -cache.params.a
        return cache.normalization * int_polynomial_cutoff(x, -cache.params)
    elseif cache.params.a < x < cache.params.b
        return one(T) - cache.normalization * int_polynomial_cutoff(x, cache.params)
    end
end

"""
    χ_der(x, cache::IntegrationCache)

Evaluate the derivative of the cutoff function `χ` at the point `x`.

# Input arguments

  - x: point at which the derivative of the cutoff function is evaluated.
  - cache: see [`IntegrationCache`](@ref).

# Returns

  - The value of the derivative of the cutoff function `χ` at `x`.
"""
function χ_der(x::T, cache::IntegrationCache) where {T}
    if -cache.params.b < x < -cache.params.a
        return cache.normalization * polynomial_cutoff(x, -cache.params)
    elseif cache.params.a < x < cache.params.b
        return -cache.normalization * polynomial_cutoff(x, cache.params)
    else
        return zero(T)
    end
end

"""
    Yε(x, cache::IntegrationCache)

Evaluate the cutoff function `Yε` at the point `x` (`C^∞` function).

# Input arguments

  - x: point at which the cutoff function is evaluated.
  - cache: see [`IntegrationCache`](@ref).

# Returns

  - The value of the cutoff function `Yε` at `x`.
"""
function Yε(x::T, cache::IntegrationCache) where {T}
    if x >= cache.params.b
        return zero(T)
    elseif zero(T) <= x <= cache.params.a
        return one(T)
    else
        return one(T) - cache.normalization * int_polynomial_cutoff(x, cache.params)
    end
end

"""
    Yε_1st_der(x, cache::IntegrationCache)

Evaluate the derivative of the cutoff function `Yε` at the point `x`.

# Input arguments

  - x: point at which the derivative of the cutoff function is evaluated.
  - cache: see [`IntegrationCache`](@ref).

# Returns

  - The value of the derivative of the cutoff function `Yε` at `x`.
"""
Yε_1st_der(x::T, cache::IntegrationCache) where {T} = cache.params.a < x < cache.params.b ?
                                                      -cache.normalization * polynomial_cutoff(x, cache.params) : zero(T)

"""
    Yε_2nd_der(x, cache::IntegrationCache)

Evaluate the 2nd order derivative of the cutoff function `Yε` at the point `x`.

# Input arguments

  - x: point at which the 2nd order derivative of the cutoff function is evaluated.
  - cache: see [`IntegrationCache`](@ref).

# Returns

  - The value of the 2nd order derivative of the cutoff function `Yε` at `x`.
"""
Yε_2nd_der(x::T, cache::IntegrationCache) where {T} = cache.params.a < x < cache.params.b ?
                                                      -cache.normalization * polynomial_cutoff_derivative(x, cache.params) :
                                                      zero(T)
