"""
Coeficients R_j^n
"""
function weights(N, j)

    y = -(-1)^j * π / N^2

    for m ∈ 1:(N - 1)
        y = y - 2 * pi / (N * m) * cos(m * j * pi / N)
    end

    y
end

"""
Parametrization of the curve (Circle shape)
"""
function curve_circle(t, R; pos_x=0.0, pos_y=0.0)
    x1 = R * cos(t) + pos_x
    x2 = R * sin(t) + pos_y
    x1, x2
end

"""
Derivative of the curve (Circle shape)
"""
function dcurve_circle(t, R)
    x1 = -R * sin(t)
    x2 = R * cos(t)
    x1, x2
end


"""
Second order derivative of the curve (Circle shape)
"""
function ddcurve_circle(t, R)
    x1 = -R * cos(t)
    x2 = -R * sin(t)
    x1, x2
end

"""
Kernel function M1
"""
function M1_circle(k, t1, t2, R, pos_x, pos_y)
    x11, x12 = curve_circle(t1, R; pos_x=pos_x, pos_y=pos_y)
    x21, x22 = curve_circle(t2, R; pos_x=pos_x, pos_y=pos_y)

    x1 = x11 - x21
    x2 = x12 - x22

    dy1, dy2 = dcurve_circle(t2, R)

    r = sqrt(x1^2 + x2^2)
    dr = sqrt(dy1^2 + dy2^2)

    -1 / 2π * Bessels.besselj(0, k * r) * dr
end

"""
Kernel function M2
"""
function M2_circle(k, t1, t2, R, pos_x, pos_y)
    x11, x12 = curve_circle(t1, R; pos_x=pos_x, pos_y=pos_y)
    x21, x22 = curve_circle(t2, R; pos_x=pos_x, pos_y=pos_y)

    x1 = x11 - x21
    x2 = x12 - x22

    dy1, dy2 = dcurve_circle(t2, R)

    r = sqrt(x1^2 + x2^2)
    dr = sqrt(dy1^2 + dy2^2)

    if t1 == t2
        return (im / 2 - eulergamma / π - 1 / π * log(k / 2 * dr)) * dr
    else
        return im / 2 * Bessels.besselh(0, 1, k * r) * dr - M1_circle(k, t1, t2, R, pos_x, pos_y) * log(4 * sin((t1 - t2) / 2)^2)
    end
end

"""
Find the closest center to a given point (xt1, xt2) from a list of centers.
"""
function find_closest_center(xt1, xt2, centers)
    closest_idx = nothing
    min_dist = Inf
    @inbounds for (i, (cx, cy)) ∈ enumerate(centers)
        dist = (xt1 - cx)^2 + (xt2 - cy)^2
        if dist < min_dist
            min_dist = dist
            closest_idx = i
        end
    end
    return closest_idx
end
