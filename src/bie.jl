# BIE solver for the Helmholtz equation in 2D, using the single-layer potential representation.

"""
    generate_SL_mat(k, N, R, pos_x, pos_y)

Generates the matrix for the single-layer potential representation of the Helmholtz equation in 2D, using a circular boundary.
"""
function generate_SL_mat(k, N, R, pos_x, pos_y)
    t = range(0; stop=(2π - π / N), length=(2 * N)) # quadrature points

    A = zeros(ComplexF64, 2 * N, 2 * N)
    for j ∈ 1:(2 * N)
        for l ∈ 1:(2 * N)
            A[j, l] = weights(N, abs(j - l)) * M1_circle(k, t[j], t[l], R, pos_x, pos_y) +
                      π / N * M2_circle(k, t[j], t[l], R, pos_x, pos_y)
        end
    end

    return SVector{2 * N}(t), SMatrix{2 * N, 2 * N}(A)
end

"""
    generate_rhs(N, t, fft_cache, R, pos_x, pos_y)

Generates the right-hand side for the single-layer potential representation of the Helmholtz equation in 2D, using a circular boundary.
"""
function generate_rhs(N, t, fft_cache, R, pos_x, pos_y)

    params, fs, cache = fft_cache

    b = zeros(ComplexF64, 2 * N)
    for j ∈ 1:(2 * N)
        x11, x12 = curve_circle(t[j], R; pos_x=pos_x, pos_y=pos_y)
        b[j] = 2 * eval_qp_green_fourier_series_eff_smooth((x11, x12), params, fs, cache; nb_terms=100)
    end

    return SVector{2 * N}(b)
end

"""
    generate_rhs_nufft(N, t, fft_cache, R, pos_x, pos_y, fourier_coeffs, idx)

Generates the right-hand side for the single-layer potential representation of the Helmholtz equation in 2D, using a circular boundary, with the Fourier coefficients provided as input (computed using a NUFFT).
"""
function generate_rhs_nufft(N, t, fft_cache, R, pos_x, pos_y, fourier_coeffs, idx)

    params, fs, cache = fft_cache

    nufft_coeffs = fourier_coeffs[idx:(idx + 2 * N - 1)]

    b = zeros(ComplexF64, 2 * N) # rhs
    for j ∈ 1:(2 * N)
        x11, x12 = curve_circle(t[j], R; pos_x=pos_x, pos_y=pos_y)
        b[j] = 2 * eval_qp_green_NUFFT_smooth((x11, x12), params, nufft_coeffs[j], cache; nb_terms=100)
    end

    return SVector{2 * N}(b)

end

"""
    evaluate_potential_BIE(k, N, xt1, xt2, phi, bd_pts)

Evaluates the single-layer potential at a target point (xt1, xt2) given the density phi and the boundary points bd_pts.
"""
function evaluate_potential_BIE(k, N, xt1, xt2, phi, bd_pts)
    y = zero(ComplexF64)

    @inbounds for j ∈ 1:(2 * N)
        x1, x2 = bd_pts[j]
        r = sqrt((xt1 - x1)^2 + (xt2 - x2)^2)

        y += Bessels.besselh(0, 1, k * r) * phi[j]
    end
    y
end
