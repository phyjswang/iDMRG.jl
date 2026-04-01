using TensorOperations
using KrylovKit
using MatrixAlgebraKit
using LinearAlgebra

# basic tensors
# -------------
sx = [0. 1.; 1. 0.] ./ 2
sz = [1. 0.; 0. -1.] ./ 2
id = [1. 0.; 0. 1.]

function add2site(hl_old::Matrix{ComplexF64}, hr_old::Matrix{ComplexF64}, szl_old::Matrix{ComplexF64}, szr_old::Matrix{ComplexF64}, D::Int64, g::Float64)
    id_old = Matrix(I, size(hl_old))
    @tensor idl_new[:] := id_old[-1,-3] * id[-2,-4]
    @tensor idr_new[:] := id[-1,-3] * id_old[-2,-4]

    @tensor hl_new[:] := hl_old[-1,-3] * id[-2,-4] - szl_old[-1,-3] * sz[-2,-4] - g * id_old[-1,-3] * sx[-2,-4]
    @tensor hr_new[:] := id[-1,-3] * hr_old[-2,-4] - sz[-1,-3] * szr_old[-2,-4] - g * sx[-1,-3] * id_old[-2,-4]
    @tensor szl_new[:] := id_old[-1,-3] * sz[-2,-4]
    @tensor szr_new[:] := sz[-1,-3] * id_old[-2,-4]

    # total Hamiltonian
    # -----------------
    @tensor htot[:] := hl_new[-1,-2,-5,-6] * idr_new[-3,-4,-7,-8] +
                       idl_new[-1,-2,-5,-6] * hr_new[-3,-4,-7,-8] +
                       id_old[-1,-5] * sz[-2,-6] * sz[-3,-7] * id_old[-4,-8]

    # ground state
    # ------------
    vals, vecs, _ = eigsolve(
        x -> tensorcontract(htot, (-1,-2,-3,-4,1,2,3,4), x, (1,2,3,4)),
        rand(size(htot)[5:8]...),
        1,
        :SR
    )
    gs = vecs[1]

    # reduced density matrix and rotation matrix
    # ----------------------------------------------
    @tensor rρl[:] := gs[-1,-2,1,2] * conj(gs)[-3,-4,1,2]
    _, Vl, _ = eigh_trunc(
        reshape(rρl, Int64(√prod(size(rρl))), :);
        trunc = truncrank(D)
    )
    Vl = reshape(Vl, (size(rρl,3), size(rρl,4), :))

    Vr = permutedims(Vl, (2,1,3)) # due to reflection symmetry

    @tensor hl_new_update[:] := conj(Vl)[1,2,-1] * hl_new[1,2,3,4] * Vl[3,4,-2]
    @tensor hr_new_update[:] := conj(Vr)[1,2,-1] * hr_new[1,2,3,4] * Vr[3,4,-2]
    @tensor szl_new_update[:] := conj(Vl)[1,2,-1] * szl_new[1,2,3,4] * Vl[3,4,-2]
    @tensor szr_new_update[:] := conj(Vr)[1,2,-1] * szr_new[1,2,3,4] * Vr[3,4,-2]

    return vals[1], hl_new_update, hr_new_update, szl_new_update, szr_new_update
end

function main(Lmax::Int64, D::Int64; g::Float64 = 0.5)
    # initialization
    # --------------
    hl = ComplexF64.(- g * sx)
    hr = hl
    szl = ComplexF64.(sz)
    szr = szl
    imax = div(Lmax, 2) - 1
    lsgseps = zeros(imax)

    for i in 1:imax
        gse, hl, hr, szl, szr = add2site(hl, hr, szl, szr, D, g)
        lsgseps[i] = real(gse / (2 * (i+1)))
    end

    return lsgseps
end

lsD = 2 .^ (2:5)
lslse = [main(64, D) for D in lsD]

# ED results at the critical point
# ---------------------
lsL_ed = [16,20,32,40,64]
lse_ed = [-1.2510242438, -1.255389856, -1.2620097863, -1.264235845, -1.267593439] ./ 4

lsidx = [div(L,2)-1 for L in [16,20,32,40,64]]
lsrelerr = [abs.((lse[lsidx] .- lse_ed) ./ lse_ed) for lse in lslse]

# plot
# ----
using CairoMakie

f=Figure(fontsize = 18)

ax=Axis(
    f[1,1],
    yscale = log10,
    xlabel = L"L",
    ylabel = "Rel. err. of GSE per site",
    xticks = lsL_ed,
)

for (i,D) in enumerate(lsD)
    scatterlines!(
        ax,
        lsL_ed,
        lsrelerr[i],
        label = "$D"
    )
end

axislegend(ax,L"D", position = :rc, titleposition = :left)

save("result.pdf", f)

f
