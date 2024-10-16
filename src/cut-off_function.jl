# Build the cut-off function χ and Yε and their derivatives from paper [1].

"""
    χ(x, c̃, c, n)

Build the cutoff function χ.
Input arguments:

  - x: point at which the cut-off function is evaluated
  - c̃: parameter of the cut-off function
  - c: parameter of the cut-off function
  - n: order of the polynomial

Returns the value of the cut-off function at x.
"""
function χ(x, c, c̃, n)

    c₁ = c
    c₂ = (c + c̃) / 2

    poly_left(x) = (x + c₁)^n * (x + c₂)^n
    poly_right(x) = (x - c₁)^n * (x - c₂)^n

    if abs(x) >= c₂
        return 0
    elseif abs(x) <= c₁
        return 1
    elseif -c₂ < x < -c₁
        int_pol_left(_x) = quadgk(poly_left, -c₂, _x)[1]

        integral_left = quadgk(poly_left, -c₂, -c₁)[1]
        cst_left = 1 / integral_left

        return cst_left * int_pol_left.(x)
    elseif c₁ < x < c₂
        int_pol_right(_x) = quadgk(poly_right, c₁, _x)[1]

        integral_right = quadgk(poly_right, c₁, c₂)[1]
        cst_right = 1 / integral_right

        return 1 - cst_right * int_pol_right.(x)
    else
        @error "Problem with the given point in χ"
    end
end

"""
    χ_der(x, c̃, c, n)

Build the derivative of the cut-off function χ.
Input arguments:

  - x: point at which the derivative of the cut-off function is evaluated
  - c̃: parameter of the cut-off function
  - c: parameter of the cut-off function
  - n: order of the polynomial

Returns the value of the derivative of the cut-off function at x.
"""
function χ_der(x, c, c̃, n)

    c₁ = c
    c₂ = (c + c̃) / 2

    poly_left(x) = (x + c₁)^n * (x + c₂)^n
    poly_right(x) = (x - c₁)^n * (x - c₂)^n

    if abs(x) >= c₂
        return 0
    elseif abs(x) <= c₁
        return 0
    elseif -c₂ < x < -c₁
        integral_left = quadgk(poly_left, -c₂, -c₁)[1]
        cst_left = 1 / integral_left

        return cst_left * poly_left.(x)
    elseif c₁ < x < c₂
        integral_right = quadgk(poly_right, c₁, c₂)[1]
        cst_right = 1 / integral_right

        return -cst_right * poly_right.(x)
    else
        @error "Problem with the given point in χ_der'"
    end
end

"""
    Yε(x, ε, n)

Build the cut-off function Yε.
Input arguments:

  - x: point at which the cut-off function is evaluated
  - ε: parameter of the cut-off function
  - n: order of the polynomial

Returns the value of the cut-off function at x.
"""
function Yε(x, ε, n)

    poly(x) = (x - ε)^n * (x - 2 * ε)^n

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 1
    elseif ε < x < 2 * ε
        int_pol(_x) = quadgk(poly, ε, _x)[1]

        integral = quadgk(poly, ε, 2 * ε)[1]
        cst = 1 / integral

        return 1 - cst * int_pol.(x)
    else
        @error "error treating x in Yε"
    end
end

"""
    Yε_1st_der(x, ε, n)

Build the derivative of the cut-off function Yε.
Input arguments:

  - x: point at which the derivative of the cut-off function is evaluated
  - ε: parameter of the cut-off function
  - n: order of the polynomial

Returns the value of the derivative of the cut-off function at x.
"""
function Yε_1st_der(x, ε, n)

    poly(x) = (x - ε)^n * (x - 2 * ε)^n

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 0
    elseif ε < x < 2 * ε
        integral = quadgk(poly, ε, 2 * ε)[1]

        cst = 1 / integral
        return -cst * poly.(x)
    else
        @error "error treating x in Yε_1st_der"
    end
end

"""
    Yε_2nd_der(x, ε, n)

Build the 2nd derivative of the cut-off function Yε.
Input arguments:

  - x: point at which the 2nd derivative of the cut-off function is evaluated
  - ε: parameter of the cut-off function
  - n: order of the polynomial

Returns the value of the 2nd derivative of the cut-off function at x.
"""
function Yε_2nd_der(x, ε, n)

    poly(x) = n * (x - ε)^(n - 1) * (x - 2 * ε)^n + n * (x - ε)^n * (x - 2 * ε)^(n - 1)

    poly_primitive(x) = (x - ε)^n * (x - 2 * ε)^n

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 0
    elseif ε < x < 2 * ε
        integral = quadgk(poly_primitive, ε, 2 * ε)[1]

        cst = 1 / integral
        return -cst * poly.(x)
    else
        @error "error treating x in Yε_2nd_der"
    end
end
