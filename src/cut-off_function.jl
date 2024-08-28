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

    g_left(x, p) = (-c₁ - x)^8 * (-c₂ - x)^8
    g_right(x, p) = (c₁ - x)^8 * (c₂ - x)^8

    if abs(x) >= c₁
        return 0
    elseif abs(x) <= c₂
        return 1
    elseif x < -c₂ && x > -c₁
        prob1 = IntegralProblem(g_left, (-c₁, -c₂))
        pol_left(x1) = solve(IntegralProblem(g_left, (-c₁, x1)), HCubatureJL()).u

        integral_left = solve(prob1, HCubatureJL()).u
        cst_left = 1 / integral_left

        return cst_left * pol_left.(x)
    elseif x > c₂ && x < c₁
        prob1 = IntegralProblem(g_right, (c₂, c₁))
        pol_right(x1) = solve(IntegralProblem(g_right, (x1, c₁)), HCubatureJL()).u

        integral_right = solve(prob1, HCubatureJL()).u
        cst_right = 1 / integral_right

        return cst_right * pol_right.(x)
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

    g_left(x, p) = (-c₁ - x)^8 * (-c₂ - x)^8
    g_right(x, p) = (c₁ - x)^8 * (c₂ - x)^8

    if abs(x) >= c₁
        return 0
    elseif abs(x) <= c₂
        return 0
    elseif x < -c₂ && x > -c₁
        prob = IntegralProblem(g_left, (-c₁, -c₂))

        integral_left = solve(prob, HCubatureJL()).u
        cst_left = 1 / integral_left
        return cst_left * g_left.(x, 0)
    elseif x > c₂ && x < c₁
        prob = IntegralProblem(g_right, (c₂, c₁))

        integral_right = solve(prob, HCubatureJL()).u
        cst_right = 1 / integral_right
        return cst_right * g_right.(x, 0)
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

    g(x, p) = (ε - x)^8 * (2 * ε - x)^8

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 1
    elseif ε < x < 2 * ε
        prob = IntegralProblem(g, (ε, 2 * ε))
        pol(x) = solve(IntegralProblem(g, (x, 2 * ε)), HCubatureJL()).u

        integral = solve(prob, HCubatureJL()).u
        cst = 1 / integral
        return cst * pol.(x)
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

    g(x, p) = (ε - x)^8 * (2 * ε - x)^8

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 0
    elseif ε < x < 2 * ε
        prob = IntegralProblem(g, (ε, 2 * ε))

        integral = solve(prob, HCubatureJL()).u
        cst = 1 / integral
        return cst * g.(x, 0)
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

    g_primitive(x, p) = (ε - x)^8 * (2 * ε - x)^8

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 0
    elseif ε < x < 2 * ε
        prob = IntegralProblem(g_primitive, (ε, 2 * ε))

        integral = solve(prob, HCubatureJL()).u
        cst = 1 / integral
        return cst * g.(x)
    else
        @error "error treating x in Yε''"
    end
end
