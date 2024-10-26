using Test
using QPGreen

@testset "Ewald method" begin
    X = 0.0
    Y = 0.01 * 2π
    a, M₁, M₂, N, β, k, d = (2, 3, 2, 7, √2 / (2 * π), 2 / (2 * π), 2 * π)
    csts = (a, M₁, M₂, N, β, k, d)

    # Test to match parameter from the paper (Linton, 1998)
    @test (X / d == 0.0, Y / d == 0.01, k * d == 2, β * d == √2)

    res_ewald = QPGreen.ewald([X, Y], csts)
    @test isapprox(res_ewald, -0.4595298795 - 0.3509130869im)

    a, M₁, M₂, N, β, k, d = (2, 2, 1, 4, √2 / (2 * π), 2 / (2 * π), 2 * π)
    csts = (a, M₁, M₂, N, β, k, d)

    # Test to match parameter from the paper (Linton, 1998)
    @test (X / d == 0.0, Y / d == 0.01, k * d == 2, β * d == √2)

    res_ewald = QPGreen.ewald([X, Y], csts)
    @test isapprox(res_ewald, -0.4595298795 - 0.3509130869im, atol=1e-6)
end
