# Tools for the package

"""
"""
gen_grid_FFT(x, y, N::Integer) = ndgrid((-x):(1 / (N - 1)):x, (-y):(1 / (N - 1)):y)
