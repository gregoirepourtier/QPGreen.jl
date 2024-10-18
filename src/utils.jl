# Tools for the package

"""
    gen_grid_FFT(x, y, N)

Generate a uniform rectangular grid of 2N*2N small rectangles.

Input arguments:

  - x: half-length of the grid in the x-direction
  - y: half-length of the grid in the y-direction
  - N: number of small rectangles in each direction

Returns the meshgrid of the grid.
"""
gen_grid_FFT(x, y, N::Integer) = ndgrid(range(-x, x - x / N; length=2 * N), range(-y, y - y / N; length=2 * N))

"""
    get_grid_pts(grid_X, grid_Y, total_pts)

Get the set of points of the grid.

Input arguments:

  - grid_X: meshgrid in the x-direction
  - grid_Y: meshgrid in the y-direction
  - total_pts: total number of points in the grid

Returns the set of points of the grid as a 2D array of size (total_pts, 2).
"""
function get_grid_pts(grid_X, grid_Y, total_pts)
    set_of_pt_grid = zeros(total_pts, 2)
    for i ∈ 1:total_pts
        set_of_pt_grid[i, 1] = grid_X[i]
        set_of_pt_grid[i, 2] = grid_Y[i]
    end

    set_of_pt_grid
end

"""
    quad(f::T, x, a, b)

Change of variable to compute the integral of a function with a Gauss Quadrature Rule.

Input arguments:

  - f: function to integrate
  - x: quadrature points
  - a: lower bound of the interval
  - b: upper bound of the interval

Returns the value of the integral on the interval [a, b].
"""
function quad_CoV(f::T, x, a, b) where {T}
    half_diff = (b - a) / 2
    mid_point = (a + b) / 2
    return half_diff * f(half_diff * x + mid_point)
end
