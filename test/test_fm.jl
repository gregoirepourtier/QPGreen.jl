Pkg.activate("test/Project.toml")
using Test, GreenFunction, LinearAlgebra, GLMakie, SpecialFunctions, FFTW

X, Y = (0.01π, 0.01)
α, c, c̃, k, ε = (0.3, 0.6, 1.0, √10, 0.1)
params = (α, c, c̃, k)

res_eig = GreenFunction.green_function_eigfct_exp((X, Y), k, α; nb_terms=1000);
res_img = GreenFunction.green_function_img_exp((X, Y), k, α; nb_terms=200000);


begin
    χ_der(x) = GreenFunction.build_χ_der(x, c̃, c)
    Yε(x) = GreenFunction.build_Yε(x, ε)
    Yε_der(x) = GreenFunction.build_Yε_der(x, ε)
    Yε_der_2nd(x) = GreenFunction.build_Yε_der_2nd(x, ε)
end;

for i in [10, 20, 32, 40, 50, 64, 80]
    @time preparation_result = GreenFunction.fm_method_preparation(params, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=i);
    res_fm = GreenFunction.fm_method_calculation((X,Y), params, preparation_result, Yε; α=α, k=k, nb_terms=32)
    println("Grid size: ", i, " and res: ", res_fm)
end

# preparation_result = GreenFunction.fm_method_preparation(params, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=32);
# GreenFunction.fm_method_calculation((X,Y), params, preparation_result, Yε; α=α, k=k, nb_terms=32)
