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
params = (alpha=0.3, k=1.0, c=0.6, c_tilde=1.0, epsilon=0.45, order=8)
nothing # hide
``` 
The parameters are defined as follows:
- `alpha` is the quasiperiod, 
- `k` is the wavenumber, 
- `c` and `c_tilde` are the coefficients introduced in [Zhang2018](@cite) that define the vertical bounds of the computational domains used in the FFT-based algorithm, determining the regions where interpolation and Fourier-based approximations are valid (consider $x=(x_1, x_2) \in \mathbb{R}^2$, then if $|x_2| > c$ the we use the eigenfunction expansion, else the FFT-based algo),
!!! note "Note on the choice of c and c_tilde"
    The following inequality must hold:
    ```math
        0 < \text{c} < \text{c_tilde}
    ```
- `epsilon` controls the support of smooth cutoff functions used to isolate and remove singularities,
- `order` denotes the regularity from the cut-off functions.

We set the grid size used to generate the 2D tensor product mesh (adjust for desired accuracy).
```@example FFT_based_algorithm
# Size of the Grid
grid_size = 128
nothing # hide
```

This method relies on two steps: a **preparation** step and an **evaluation** step. 

### Preparation step
The preparation step, independent of evaluating the function at specific points, computes the necessary Fourier coefficients and produces an interpolation object for function and gradient evaluation (if the flag is true), along with a cache of integral approximations. This step needs to be performed only once for a fixed set of parameters and grid resolution.
```@example FFT_based_algorithm
# Preparation Step
value_interpolator, grad_interpolator, cache = init_qp_green_fft(params, grid_size; grad=true)
nothing # hide
```

### Evaluation step
The evaluation step efficiently computes the Green’s function and its first derivative at any specified point within the periodic domain.
```@example FFT_based_algorithm
# Evaluation Step for any points
Z = (0.002π, 0.01)
eval_Green_function = eval_qp_green(Z, params, value_interpolator, cache; nb_terms=50)
grad_Green_function = grad_qp_green(Z, params, grad_interpolator, cache; nb_terms=50)
@show "The value of the Green function at the point $(Z) is: $(eval_Green_function)"
@show "The value of the gradient of the Green function at the point $(Z) is: $(grad_Green_function)"
```
Note here that the keyword argument `nb_terms` is used to control the number of terms in the eigenfunction expansion (in the case where the FFT-based algorithm is not used).

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
