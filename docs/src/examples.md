# Examples

We show here some examples on how to compute efficiently Quasi-periodic Green Functions for the 2D Helmholtz equation.

# Example 1: FFT-based algorithm
The first example is concerned with computing the quasi-periodic Green function for the 2D Helmholtz equation using the FFT-based algorithm (see [Zhang2018](@cite)).

```julia
using QPGreen
```

Then initialize the different required parameters and the grid size you need for the desired accuracy.
```julia
# Parameters
params = (α=0.3, k=1.0, c=0.6, c̃=1.0, ε=0.4341, order=8)

# Size of the Grid
grid_size = 128
```

This method relies on two steps: a preparation step and an evaluation step. The preparation step is used to precompute the necessary data for the evaluation step. The evaluation step is used to compute the Green function for any given point.
```julia
# Preparation Step
preparation_result, interp, cache = fm_method_preparation(params, grid_size);

# Evaluation Step for any points
Z = (0.002π, 0.0)
eval_Green_function = fm_method_calculation(Z, params, preparation_result, interp, cache; nb_terms=32)
```

# Example 2: Lattice sums algorithm

See [Linton1998](@cite)

# Example 3: Ewald summation algorithm

See [Linton1998](@cite)
