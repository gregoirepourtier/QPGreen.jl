using Test, QPGreen

X, Y = (0.002π, 0.0)
α, c, c̃, k, ε = (0.3, 0.6, 1.0, 1, 0.1)
params = (α, c, c̃, k)

res_eig = eigfunc_expansion((X, Y), k, α; nb_terms=1000)
res_img = image_expansion((X, Y), k, α; nb_terms=200000)


preparation_result = QPGreen.fm_method_preparation(params; grid_size=32);
QPGreen.fm_method_calculation((X, Y), params, preparation_result, Yε; α=α, k=k, nb_terms=32)


for i ∈ [32, 64, 80, 128]
    preparation_result = QPGreen.fm_method_preparation(params, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=i)
    res_fm = QPGreen.fm_method_calculation((X, Y), params, preparation_result, Yε; α=α, k=k, nb_terms=32)
    println("Grid size: ", i, " and res: ", res_fm)
end
