# Build the cutoff functions χ and Yε and their derivatives from paper [1].

"""
    χ(x, c̃, c, n, cache)

Build the cutoff function χ.
Input arguments:

  - x: point at which the cutoff function is evaluated
  - c̃: parameter of the cutoff function
  - c: parameter of the cutoff function
  - n: order of the polynomial
  - cache: quadrature points and weights from FastGaussQuadrature

Returns the value of the cutoff function at x.
"""
function χ(x, c₁, c₂, cst_left, cst_right, poly_left::T1, poly_right::T2) where {T1, T2}
    if abs(x) >= c₂
        return 0.0
    elseif abs(x) <= c₁
        return 1.0
    elseif -c₂ < x < -c₁
        return cst_left * poly_left(x)
    elseif c₁ < x < c₂
        return 1 - cst_right * poly_right(x)
    end
end

"""
    χ_der(x, c̃, c, n, cache)

Build the derivative of the cutoff function χ.
Input arguments:

  - x: point at which the derivative of the cutoff function is evaluated
  - c̃: parameter of the cutoff function
  - c: parameter of the cutoff function
  - n: order of the polynomial
  - cache: quadrature points and weights from FastGaussQuadrature

Returns the value of the derivative of the cutoff function at x.
"""
function χ_der(x, c₁, c₂, cst_left, cst_right, poly_left::T1, poly_right::T2) where {T1, T2}
    if -c₂ < x < -c₁
        return cst_left * poly_left(x)
    elseif c₁ < x < c₂
        return -cst_right * poly_right(x)
    else
        return 0.0
    end
end

"""
    Yε(x, ε, n, cache)

Build the cutoff function Yε.
Input arguments:

  - x: point at which the cutoff function is evaluated
  - ε: parameter of the cutoff function
  - n: order of the polynomial
  - cache: quadrature points and weights from FastGaussQuadrature

Returns the value of the cutoff function at x.
"""
function Yε(x, ε, cst, poly::T) where {T}
    if x >= 2 * ε
        return 0.0
    elseif 0 <= x <= ε
        return 1.0
    else
        return 1.0 - cst * poly(x)
    end
end

"""
    Yε_1st_der(x, ε, n, cache)

Build the derivative of the cutoff function Yε.
Input arguments:

  - x: point at which the derivative of the cutoff function is evaluated
  - ε: parameter of the cutoff function
  - n: order of the polynomial
  - cache: quadrature points and weights from FastGaussQuadrature

Returns the value of the derivative of the cutoff function at x.
"""
Yε_1st_der(x, ε, cst, poly::T) where {T} = ε < x < 2 * ε ? -cst * poly(x) : 0.0

"""
    Yε_2nd_der(x, ε, n, cache)

Build the 2nd derivative of the cutoff function Yε.
Input arguments:

  - x: point at which the 2nd derivative of the cutoff function is evaluated
  - ε: parameter of the cutoff function
  - n: order of the polynomial
  - cache: quadrature points and weights from FastGaussQuadrature

Returns the value of the 2nd derivative of the cutoff function at x.
"""
Yε_2nd_der(x, ε, cst, poly::T) where {T} = ε < x < 2 * ε ? -cst * poly(x) : 0.0
