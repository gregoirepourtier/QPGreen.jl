using QPGreen
using QuadGK

ε = 0.4341
order = 8

params_Yε = QPGreen.IntegrationParameters(ε, 2 * ε, order)
cache_Yε = QPGreen.IntegrationCache(params_Yε)

quadgk(x_ -> QPGreen.f₂(x_, cache_Yε), poly.a, poly.b)[1]
