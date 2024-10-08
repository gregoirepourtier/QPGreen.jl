Pkg.activate("test/Project.toml")
using Test, QPGreen, LinearAlgebra, GLMakie, SpecialFunctions, FFTW

# X, Y = (0.01π, 0.01)
# α, c, c̃, k, ε = (0.3, 0.6, 1.0, √10, 0.1)
# params = (α, c, c̃, k)
X, Y = (0.002π, 0.0)
α, c, c̃, k, ε = (0.3, 0.6, 1.0, 1, 0.1)
params = (α, c, c̃, k)

res_eig = QPGreen.green_function_eigfct_exp((X, Y), k, α; nb_terms=1000)
res_img = QPGreen.green_function_img_exp((X, Y), k, α; nb_terms=200000)


begin
    χ_der(x) = QPGreen.build_χ_der(x, c̃, c)
    Yε(x) = QPGreen.build_Yε(x, ε)
    Yε_der(x) = QPGreen.build_Yε_der(x, ε)
    Yε_der_2nd(x) = QPGreen.build_Yε_der_2nd(x, ε)
end;

preparation_result = QPGreen.fm_method_preparation(params, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=32);
QPGreen.fm_method_calculation((X,Y), params, preparation_result, Yε; α=α, k=k, nb_terms=32)


for i in [32, 64, 80, 128]
    preparation_result = QPGreen.fm_method_preparation(params, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=i);
    res_fm = QPGreen.fm_method_calculation((X,Y), params, preparation_result, Yε; α=α, k=k, nb_terms=32)
    println("Grid size: ", i, " and res: ", res_fm)
end
