# Build the cutoff functions χ and Yε and their derivatives from paper [1].

"""
    χ(x, cache)

Build the cutoff function χ.
Input arguments:

  - x: point at which the cutoff function is evaluated
  - cache: see [`IntegrationCache`](@ref)

Returns the value of the cutoff function at x.
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
    χ_der(x, cache)

Build the derivative of the cutoff function χ.
Input arguments:

  - x: point at which the derivative of the cutoff function is evaluated
  - cache: see [`IntegrationCache`](@ref)

Returns the value of the derivative of the cutoff function at x.
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
    Yε(x, cache)

Build the cutoff function Yε.
Input arguments:

  - x: point at which the cutoff function is evaluated
  - cache: see [`IntegrationCache`](@ref)

Returns the value of the cutoff function at x.
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
    Yε_1st_der(x, cache)

Build the derivative of the cutoff function Yε.
Input arguments:

  - x: point at which the derivative of the cutoff function is evaluated
  - cache: see [`IntegrationCache`](@ref)

Returns the value of the derivative of the cutoff function at x.
"""
Yε_1st_der(x::T, cache::IntegrationCache) where {T} = cache.params.a < x < cache.params.b ?
                                                      -cache.normalization * polynomial_cutoff(x, cache.params) : zero(T)

"""
    Yε_2nd_der(x, cache)

Build the 2nd derivative of the cutoff function Yε.
Input arguments:

  - x: point at which the 2nd derivative of the cutoff function is evaluated
  - cache: see [`IntegrationCache`](@ref)

Returns the value of the 2nd derivative of the cutoff function at x.
"""
Yε_2nd_der(x::T, cache::IntegrationCache) where {T} = cache.params.a < x < cache.params.b ?
                                                      -cache.normalization * polynomial_cutoff_derivative(x, cache.params) :
                                                      zero(T)
