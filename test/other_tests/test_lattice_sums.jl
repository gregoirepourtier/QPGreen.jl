using Test
using QPGreen

X, Y = (0.0, 0.01 * 2π)
# X, Y = (0.0, 0.002)
r, θ = (√(X^2 + Y^2), atan(Y, X))

β, k, d, M, L = (√2 / (2 * π), 2 / (2 * π), 2 * π, 80, 4)
# β, k, d, M, L = (0.3, 1, 2 * π, 80, 4)
csts = (β, k, d, M, L)

# Test to match parameter from the paper (Linton, 1998)
(X / d == 0.0, Y / d == 0.01, k * d == 2, β * d == √2, r < d)

Sl = QPGreen.lattice_sums_preparation(csts);
eval_ls = QPGreen.lattice_sums_calculation((X, Y), csts, Sl; nb_terms=100)

@test eval_ls == (-0.4595073481223693 - 0.35087920819684576 * im)
