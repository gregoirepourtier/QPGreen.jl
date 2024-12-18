using Test
using QPGreen

@testset "Expansions" begin
    include("other_tests/test_expansions.jl")
end

@testset "Ewald method" begin
    include("other_tests/test_ewald.jl")
end

@testset "FFT based algorithm" begin
    include("test_fft_method_eval.jl")
end

@testset "Lattice Sums" begin
    include("other_tests/test_lattice_sums.jl")
end
