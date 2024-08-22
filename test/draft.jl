# Some experimental code


### Test Cut-off function ###
using FastGaussQuadrature
function g_1(x, c, c̃)

    c₁ = (c + c̃) / 2
    c₂ = c

    # 2 ≠ options here  
    g_left(x) = (-c₁ - x)^5 * (-c₂ - x)^5 # (c₁ - x)^5 * (c₂ - x)^5 
    g_right(x) = (c₁ - x)^5 * (c₂ - x)^5 # (-c₁ - x)^5 * (-c₂ - x)^5

    ξ, w = gausslegendre(5)

    if abs(x) >= c₁
        return 0
    elseif abs(x) <= c₂
        return 1
    elseif x < -c₂ && x > -c₁
        integral_left = dot(w, GreenFunction.quad.(g_left, ξ, -c₁, -c₂))
        cst_left = 1 / integral_left
        return cst_left * dot(w, GreenFunction.quad.(g_left, ξ, -c₁, x))
    elseif x > c₂ && x < c₁
        integral_right = dot(w, GreenFunction.quad.(g_right, ξ, c₂, c₁))
        cst_right = 1 / integral_right
        return cst_right * dot(w, GreenFunction.quad.(g_right, ξ, x, c₁))
    end
end

function g_1_der(x, c, c̃)

    c₁ = (c + c̃) / 2
    c₂ = c

    # 2 ≠ options here  
    g_left(x) = (-c₁ - x)^5 * (-c₂ - x)^5 # (c₁ - x)^5 * (c₂ - x)^5 
    g_right(x) = (c₁ - x)^5 * (c₂ - x)^5 # (-c₁ - x)^5 * (-c₂ - x)^5

    ξ, w = gausslegendre(5)

    if abs(x) >= c₁
        return 0
    elseif abs(x) <= c₂
        return 0
    elseif x < -c₂ && x > -c₁
        integral_left = dot(w, GreenFunction.quad.(g_left, ξ, -c₁, -c₂))
        cst_left = 1 / integral_left
        return cst_left * g_left.(x)
    elseif x > c₂ && x < c₁
        integral_right = dot(w, GreenFunction.quad.(g_right, ξ, c₂, c₁))
        cst_right = 1 / integral_right
        return cst_right * g_right.(x)
    end
end

function g_1_der_2nd(x, c, c̃)

    c₁ = (c + c̃) / 2
    c₂ = c

    # 2 ≠ options here
    g_left(x) = -5 * (-c₁ - x)^4 * (-c₂ - x)^5 - 5 * (-c₁ - x)^5 * (-c₂ - x)^4
    g_right(x) = -5 * (c₁ - x)^4 * (c₂ - x)^5 - 5 * (c₁ - x)^5 * (c₂ - x)^4

    # 2 ≠ options here Primitive to compute constants
    g_left_primitive(x) = (-c₁ - x)^5 * (-c₂ - x)^5 # (c₁ - x)^5 * (c₂ - x)^5 
    g_right_primitive(x) = (c₁ - x)^5 * (c₂ - x)^5 # (-c₁ - x)^5 * (-c₂ - x)^5

    ξ, w = gausslegendre(5)

    if abs(x) >= c₁
        return 0
    elseif abs(x) <= c₂
        return 0
    elseif x < -c₂ && x > -c₁
        integral_left = dot(w, GreenFunction.quad.(g_left_primitive, ξ, -c₁, -c₂))
        cst_left = 1 / integral_left
        return cst_left * g_left.(x)
    elseif x > c₂ && x < c₁
        integral_right = dot(w, GreenFunction.quad.(g_right_primitive, ξ, c₂, c₁))
        cst_right = 1 / integral_right
        return cst_right * g_right.(x)
    end
end

c̃ = 3.0
c = 1.0
# x = collect((-c̃ - 1.0):0.01:(c̃ + 1.0));
x = collect(-3.0:0.01:3.0);
y = g_1.(x, c, c̃);


f = Figure()

ax = Axis(f[1, 1])

lines!(ax, x, y; color=:black)
f
empty!(ax)




begin
    function bump(x)
        if -1 < x < 1
            return exp(-1 / (1 - x^2))
        else
            return 0
        end
        # if x > 0
        #     return exp(-1 / x)
        # else
        #     return 0
        # end
    end

    function bump_der(x)
        if -1 < x < 1
            return -2 * x * exp(-1 / (1 - x^2)) / (1 - x^2)^2
        else
            return 0
        end
    end

    function bump2(x)
        if x > 0
            return exp(-1 / x)
        else
            return 0
        end
    end
    function bump2_der(x)
        if x > 0
            return 1 / x^2 * exp(-1 / x)
        else
            return 0
        end
    end
    function bump2_der_2nd(x)
        if x > 0
            return (1 - 2 * x) / x^4 * exp(-1 / x)
        else
            return 0
        end
    end

    g(x) = bump2(x) / (bump2(x) + bump2(1 - x))
    g_der(x) = bump2_der(x) / (bump2_der(x) + bump2_der(1 - x))
    g_der_2nd(x) = bump2_der_2nd(x) / (bump2_der_2nd(x) + bump2_der_2nd(1 - x))


    x = collect(-2.0:0.01:2.0)
    y = bump2_der_2nd.(x) # g.((x .+ 1) ./ (-2 + 1))
    fig = Figure()

    ax = Axis(fig[1, 1]; xticks=(-2.0:0.5:2.0))
    lines!(ax, x, y)
    fig
end




function g_1(x, ε)

    # 2 ≠ options here  
    g(x) = (ε - x)^5 * (2 * ε - x)^5 # x^5 * (-2*ε - x)^5 

    ξ, w = gausslegendre(5)

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 1
    elseif ε < x < 2 * ε
        integral = dot(w, GreenFunction.quad.(g, ξ, ε, 2 * ε))
        cst = 1 / integral
        return cst * dot(w, GreenFunction.quad.(g, ξ, x, 2 * ε))
    else
        @error "x is out of bounds"
    end
end

function g_1_der(x, ε)

    # 2 ≠ options here  
    g(x) = (ε - x)^5 * (2 * ε - x)^5 # x^5 * (-2*ε - x)^5 

    ξ, w = gausslegendre(5)

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 0
    elseif ε < x < 2 * ε
        integral = dot(w, GreenFunction.quad.(g, ξ, ε, 2 * ε))
        cst = 1 / integral
        return cst * g.(x)
    else
        @error "x is out of bounds"
    end
end

function g_1_der_2nd(x, ε)

    g(x) = -5 * (ε - x)^4 * (2 * ε - x)^5 - 5 * (ε - x)^5 * (2 * ε - x)^4

    # 2 ≠ options here  
    g_primitive(x) = (ε - x)^5 * (2 * ε - x)^5 # x^5 * (-2*ε - x)^5 

    ξ, w = gausslegendre(5)

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 0
    elseif ε < x < 2 * ε
        integral = dot(w, GreenFunction.quad.(g_primitive, ξ, ε, 2 * ε))
        cst = 1 / integral
        return cst * g.(x)
    else
        @error "x is out of bounds"
    end
end

ε = 0.1
x = collect(0.0:0.001:1.0);
y = g_1_der_2nd.(x, ε);


f = Figure()

ax = Axis(f[1, 1]; xticks=(0.0:0.1:1.0))

lines!(ax, x, y; color=:black)
f
empty!(ax)
