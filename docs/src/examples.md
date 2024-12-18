# Examples

```@meta
CurrentModule = QPGreen
```

We show here some examples on how to compute efficiently Quasi-periodic Green Functions and its first derivatives for the 2D Helmholtz equation. The main focus is on the FFT-based algorithm, but we also provide examples for the lattice sums and Ewald summation algorithms.

# Example 1: FFT-based algorithm
The first example is concerned with computing the quasi-periodic Green function for the 2D Helmholtz equation using the **FFT-based algorithm** (based on the article [Zhang2018](@cite)).

First, load the package, using the following command:
```@example FFT_based_algorithm
using QPGreen
```

Then initialize the different parameters, as a NamedTuple, that define your problems. We have
```@example FFT_based_algorithm
# Parameters
params = (α=0.3, k=1.0, c=0.6, c̃=1.0, ε=0.45, order=8)
``` 
with the following parameters:
- `α` is the quasiperiod, 
- `k` is the wavenumber, 
- `c` and `c̃` are coefficients introduced in the paper [Zhang2018](@cite) that are used in the smooth cut-off functions and in order to determine if we use the classical eigenfunction expansion for the approximation or the FFT-based algorithm (if you evaluate your 2D Green function in a point $x=(x_1, x_2) \in \mathbb{R}^2$, then if $|x_2| > c$ the we use the expansion, else the FFT-base algo).
!!! note "Note on the choice of c and c̃"
    The following inequality must hold:
    ```math
        0 < c < \tilde{c}
    ```
- `ε` is used in the smooth cut-off functions, and 
- `order` denote the regularity from the cut-off functions.

We set the grid size that is used to generate the 2D tensor product mesh (adjust for desired accuracy).
```@example FFT_based_algorithm
# Size of the Grid
grid_size = 128
nothing # hide
```

This method relies on two steps: a **preparation** step and an **evaluation** step. 

### Preparation step
The preparation step, which is independent of the function evaluation at a point, computes various Fourier coefficients and returns an Extrapolation object along with a cache containing integral approximations. Note that you only need to perform the preparation step once for a given set of parameters and grid size.
```@example FFT_based_algorithm
# Preparation Step
interpolation, cache = fm_method_preparation(params, grid_size)
```

### Evaluation step
The evaluation step is used to compute the Green function for any given point in the periodic domain and is very efficient.
```@example FFT_based_algorithm
# Evaluation Step for any points
Z = (0.002π, 0.01)
eval_Green_function = fm_method_calculation(Z, params, interpolation, cache; nb_terms=32)
@show "The value of the Green function at the point $(Z) is: $(eval_Green_function)"
```
Note here that the keyword argument `nb_terms` is used to control the number of terms in the eigenfunction expansion (in the case where the FFT-based algorithm is not used).

### Evaluation of the first derivatives of the Green function
In the previous section, we explained how to compute efficiently the quasi-periodic Green function. We now show how to compute the first derivatives of the Green function. The method is similar to the one used before.

Supposing that we are using the same parameters and the same grid size, we can once again use the preparation step.
```@example FFT_based_algorithm
# Preparation Step
interpolation_x1, interpolation_x2, cache = fm_method_preparation_derivative(params, grid_size);
```
Note here that we have two interpolations, one for the first derivative with respect to the first variable and the other for the second variable.

Then, we can evaluate the first derivatives of the Green function at any point in the periodic domain.
```@example FFT_based_algorithm
# Evaluation Step for any points
Z = (0.002π, 0.01)
eval_Green_function_x1 = fm_method_calculation_derivative(Z, params, interpolation_x1, interpolation_x2, cache; nb_terms=32)
```

# Example 2: Lattice sums algorithm
!!! warning "Work in progress"
    This algorithm is still a work in progress.

The second algorithm implemented is the **lattice sums algorithm** (implementation was done based on the article [Linton1998](@cite)).

Once again, load the package, using the following command:
```@example Lattice_sums_algorithm
using QPGreen
```

Then initialize the different parameters, as a NamedTuple, that define your problems. We have
```@example Lattice_sums_algorithm
X, Y = (0.0, 0.01 * 2π)

β, k, d, M, L = (√2 / (2 * π), 1 / π, 2 * π, 80, 4)
csts = (β, k, d, M, L)

Sl = QPGreen.lattice_sums_preparation(csts);
eval_ls = QPGreen.lattice_sums_calculation((X, Y), csts, Sl; nb_terms=100)
```

!!! note
    The implementation for the derivatives is missing at the moment.

# Example 3: Ewald's method
!!! warning "Work in progress"
    This algorithm is still a work in progress.

The last algorithm implemented is the **Ewald summation algorithm** (implementation was done based on the article [Linton1998](@cite)).

Load the package, using the following command:
```@example Ewald_algorithm
using QPGreen
```

Then initialize the different parameters, as a NamedTuple, that define your problems. We have
```@example Ewald_algorithm
(X, Y) = (0.0, 0.01 * 2π)

a, M₁, M₂, N, β, k, d = (2, 3, 2, 7, √2 / (2 * π), 1 / π, 2 * π)
csts = (a, M₁, M₂, N, β, k, d)

res_ewald = QPGreen.ewald([X, Y], csts)
```
!!! note
    The implementation for the derivatives is missing at the moment.
