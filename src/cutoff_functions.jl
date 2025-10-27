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
    else
        if cache.type_cutoff == :polynomial
            return one(T) - cache.normalization * int_polynomial_cutoff(abs(x), cache.params)
        else
            k = cache.params.order
            return mollifier(abs(x), cache.params, k)
        end
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
    if cache.params.a < abs(x) < cache.params.b
        if cache.type_cutoff == :polynomial
            return -sign(x) * cache.normalization * polynomial_cutoff(abs(x), cache.params)
        else
            k = cache.params.order
            return sign(x) * mollifier_derivative(abs(x), cache.params, k)
        end
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
        if cache.type_cutoff == :polynomial
            return one(T) - cache.normalization * int_polynomial_cutoff(x, cache.params)
        else
            k = cache.params.order
            return mollifier(x, cache.params, k)
        end
    end
end

# function Yε_mollifier(x::T, cache::IntegrationCache) where {T}
#     if x >= cache.params.b
#         return zero(T)
#     elseif zero(T) <= x <= cache.params.a
#         return one(T)
#     else
#         k = 2.0
#         return mollifier(x, cache.params, k)
#     end
# end

"""
    Yε_1st_der(x, cache::IntegrationCache)

Evaluate the derivative of the cutoff function `Yε` at the point `x`.

# Input arguments

  - x: point at which the derivative of the cutoff function is evaluated.
  - cache: see [`IntegrationCache`](@ref).

# Returns

  - The value of the derivative of the cutoff function `Yε` at `x`.
"""
function Yε_1st_der(x::T, cache::IntegrationCache) where {T}
    if cache.params.a < x < cache.params.b
        if cache.type_cutoff == :polynomial
            return -cache.normalization * polynomial_cutoff(x, cache.params)
        else
            k = cache.params.order
            return mollifier_derivative(x, cache.params, k)
        end
    else
        return zero(T)
    end
end

# function Yε_1st_der_mollifier(x::T, cache::IntegrationCache) where {T}
#     if cache.params.a < x < cache.params.b
#         k = 2.0
#         return mollifier_derivative(x, cache.params, k)
#     else
#         return zero(T)
#     end
# end

"""
    Yε_2nd_der(x, cache::IntegrationCache)

Evaluate the 2nd order derivative of the cutoff function `Yε` at the point `x`.

# Input arguments

  - x: point at which the 2nd order derivative of the cutoff function is evaluated.
  - cache: see [`IntegrationCache`](@ref).

# Returns

  - The value of the 2nd order derivative of the cutoff function `Yε` at `x`.
"""
function Yε_2nd_der(x::T, cache::IntegrationCache) where {T}
    if cache.params.a < x < cache.params.b
        if cache.type_cutoff == :polynomial
            return -cache.normalization * polynomial_cutoff_derivative(x, cache.params)
        else
            k = cache.params.order
            return mollifier_second_derivative(x, cache.params, k)
        end
    else
        return zero(T)
    end
end

# function Yε_2nd_der_mollifier(x::T, cache::IntegrationCache, ::Val{:mollifier}) where {T}
#     if cache.params.a < x < cache.params.b
#         k = 2.0
#         return mollifier_second_derivative(x, cache.params, k)
#     else
#         return zero(T)
#     end
# end
