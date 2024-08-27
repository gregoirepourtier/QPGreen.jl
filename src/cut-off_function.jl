"""
    build_χ(x, c̃, c)

Build the cut-off function χ.
Input arguments:

  - x: point at which the cut-off function is evaluated
  - c̃: parameter of the cut-off function
  - c: parameter of the cut-off function

Returns the value of the cut-off function at x.
"""
function build_χ(x, c̃, c)

    c₁ = (c + c̃) / 2
    c₂ = c

    g_left(x) = (-c₁ - x)^8 * (-c₂ - x)^8
    g_right(x) = (c₁ - x)^8 * (c₂ - x)^8

    ξ, w = gausslegendre(5)

    if abs(x) >= c₁
        return 0
    elseif abs(x) <= c₂
        return 1
    elseif x < -c₂ && x > -c₁
        integral_left = dot(w, quad.(g_left, ξ, -c₁, -c₂))
        cst_left = 1 / integral_left
        return cst_left * dot(w, quad.(g_left, ξ, -c₁, x))
    elseif x > c₂ && x < c₁
        integral_right = dot(w, quad.(g_right, ξ, c₂, c₁))
        cst_right = 1 / integral_right
        return cst_right * dot(w, quad.(g_right, ξ, x, c₁))
    else
        @error "Problem with the given point in χ"
    end
end

"""
    build_χ_der(x, c̃, c)

Build the derivative of the cut-off function χ.
Input arguments:

  - x: point at which the derivative of the cut-off function is evaluated
  - c̃: parameter of the cut-off function
  - c: parameter of the cut-off function

Returns the value of the derivative of the cut-off function at x.
"""
function build_χ_der(x, c̃, c)

    c₁ = (c + c̃) / 2
    c₂ = c

    g_left(x) = (-c₁ - x)^8 * (-c₂ - x)^8
    g_right(x) = (c₁ - x)^8 * (c₂ - x)^8

    ξ, w = gausslegendre(5)

    if abs(x) >= c₁
        return 0
    elseif abs(x) <= c₂
        return 0
    elseif x < -c₂ && x > -c₁
        integral_left = dot(w, quad.(g_left, ξ, -c₁, -c₂))
        cst_left = 1 / integral_left
        return cst_left * g_left.(x)
    elseif x > c₂ && x < c₁
        integral_right = dot(w, quad.(g_right, ξ, c₂, c₁))
        cst_right = 1 / integral_right
        return cst_right * g_right.(x)
    else
        @error "Problem with the given point in χ'"
    end
end

"""
    build_Yε(x, ε)

Build the cut-off function Yε.
Input arguments:

  - x: point at which the cut-off function is evaluated
  - ε: parameter of the cut-off function

Returns the value of the cut-off function at x.
"""
function build_Yε(x, ε)

    g(x) = (ε - x)^8 * (2 * ε - x)^8

    ξ, w = gausslegendre(5)

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 1
    elseif ε < x < 2 * ε
        integral = dot(w, quad.(g, ξ, ε, 2 * ε))
        cst = 1 / integral
        return cst * dot(w, quad.(g, ξ, x, 2 * ε))
    else
        @error "error treating x in Yε"
    end
end

"""
    build_Yε_der(x, ε)

Build the derivative of the cut-off function Yε.
Input arguments:

  - x: point at which the derivative of the cut-off function is evaluated
  - ε: parameter of the cut-off function

Returns the value of the derivative of the cut-off function at x.
"""
function build_Yε_der(x, ε)

    g(x) = (ε - x)^8 * (2 * ε - x)^8

    ξ, w = gausslegendre(5)

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 0
    elseif ε < x < 2 * ε
        integral = dot(w, quad.(g, ξ, ε, 2 * ε))
        cst = 1 / integral
        return cst * g.(x)
    else
        @error "error treating x in Yε'"
    end
end

"""
    build_Yε_der_2nd(x, ε)

Build the 2nd derivative of the cut-off function Yε.
Input arguments:

  - x: point at which the 2nd derivative of the cut-off function is evaluated
  - ε: parameter of the cut-off function

Returns the value of the 2nd derivative of the cut-off function at x.
"""
function build_Yε_der_2nd(x, ε)

    g(x) = -8 * (ε - x)^7 * (2 * ε - x)^8 - 8 * (ε - x)^8 * (2 * ε - x)^7

    g_primitive(x) = (ε - x)^8 * (2 * ε - x)^8

    ξ, w = gausslegendre(5)

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 0
    elseif ε < x < 2 * ε
        integral = dot(w, quad.(g_primitive, ξ, ε, 2 * ε))
        cst = 1 / integral
        return cst * g.(x)
    else
        @error "error treating x in Yε''"
    end
end
