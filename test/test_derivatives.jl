using QPGreen
using Test

function expansion_test(z, k, α)
    G_prime_eigfun = QPGreen.analytical_derivative(z, k, α; period=2π, nb_terms=100_000)

end

z = [1.0, 0.0]
k, α = (10.0, 0.3)

expansion_test(z, k, α)



@testset "Test suite expansions" begin
    z = [1.0, 2.0]
    k, α = (10.0, 0.3)

    expansion_test(z, k, α)
    # Add more tests here
end
