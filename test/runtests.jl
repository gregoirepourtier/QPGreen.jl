using Test
using QPGreen

@testset "Expansions" begin
    include("test_expansions.jl")
end

@testset "Ewald method" begin
    include("test_ewald.jl")
end

@testset "FFT based algorithm" begin
    include("test_fft_method.jl")
end

@testset "Lattice Sums" begin
    include("test_lattice_sums.jl")
end
