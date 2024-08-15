# Tools for the package

"""
"""
gen_grid_FFT(x, y, N::Integer) = ndgrid(range(-x, x; length=2 * N), range(-y, y; length=2 * N))

"""
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
"""
quad(f::T, x, a, b) where {T} = (b - a) / 2 * f((b - a) / 2 * x + (a + b) / 2)
