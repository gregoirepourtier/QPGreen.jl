using Pkg
Pkg.activate("test/Project.toml")

using QPGreen
using BenchmarkTools


params = (α=0.3, k=1.0, c=0.6, c̃=1.0, ε=0.4341, order=8)
grid_size = 16


# res_eig = 0.7685069487 + 0.1952423542im

pt = (0.002π, 0.0);
interp, cache = QPGreen.fm_method_preparation(params, grid_size);
G_x = QPGreen.fm_method_calculation(pt, params, interp, cache; nb_terms=100)

interp, cache = QPGreen.fm_method_preparation_hankel(params, grid_size);
G_x = QPGreen.fm_method_calculation_hankel(pt, params, interp, cache; nb_terms=100)