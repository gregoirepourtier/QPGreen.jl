# Tools for the package

"""
"""

gen_grid_FFT(x, y, N::Integer) = ndgrid(range(-x,x,length=2*N), range(-y,y,length=2*N))


quad(f::T, x, a, b) where {T} = (b - a) / 2 * f((b - a) / 2 * x + (a + b) / 2)
