from pathlib import Path
import numpy as np
import scipy

np.set_printoptions(precision=15)

# Assumes the script is run from the main block
output_folder = Path("include/generated")

ELE_ORDER = 8
BLKS_MAX = 1
CHEB_REDUCED_PER_BLK = ELE_ORDER // 2
DTYPE = np.double

print("Now generating hard-coded Chebyshev matrices!")
print("Element order is:", ELE_ORDER)
print("Generating up to this many fused blocks:", BLKS_MAX)
print("DataType is:", DTYPE)
print("Reduced number of points per block:", CHEB_REDUCED_PER_BLK)


def full_kron_product_truncated_scipy(x, y, z, N, Q, S):
    # did it this way so that we don't explode memory constraints
    x_arr = np.arange(x, dtype=DTYPE) / (x - 1)
    N_arr = np.arange(N, dtype=DTYPE)
    A = np.zeros((x, N), dtype=DTYPE)
    for i, x_ in enumerate(x_arr):
        for j, n_ in enumerate(N_arr):
            A[i, j] = scipy.special.eval_chebyt(n_, x_)

    y_arr = np.arange(y) / (y - 1)
    Q_arr = np.arange(Q)
    B = np.zeros((y, Q))
    for i, y_ in enumerate(y_arr):
        for j, q_ in enumerate(Q_arr):
            B[i, j] = scipy.special.eval_chebyt(q_, y_)

    z_arr = np.arange(z) / (z - 1)
    S_arr = np.arange(S)
    C = np.zeros((z, S))
    for i, z_ in enumerate(z_arr):
        for j, s_ in enumerate(S_arr):
            C[i, j] = scipy.special.eval_chebyt(s_, z_)

    UA, _, _ = np.linalg.svd(A)
    UB, _, _ = np.linalg.svd(B)
    UC, _, _ = np.linalg.svd(C)
    _ = None

    # trim them up
    UA = UA[:, :N]
    UB = UB[:, :Q]
    UC = UC[:, :S]

    UA_UB_kron = np.kron(UA, UB)
    full_kron = np.kron(UA_UB_kron, UC)

    return full_kron


def full_kron_product_truncated_scipy_2d(x, y, N, Q):
    # did it this way so that we don't explode memory constraints
    x_arr = np.arange(x) / (x - 1)
    N_arr = np.arange(N)
    A = np.zeros((x, N))
    for i, x_ in enumerate(x_arr):
        for j, n_ in enumerate(N_arr):
            A[i, j] = scipy.special.eval_chebyt(n_, x_)

    y_arr = np.arange(y) / (y - 1)
    Q_arr = np.arange(Q)
    B = np.zeros((y, Q))
    for i, y_ in enumerate(y_arr):
        for j, q_ in enumerate(Q_arr):
            B[i, j] = scipy.special.eval_chebyt(q_, y_)

    UA, _, _ = np.linalg.svd(A)
    UB, _, _ = np.linalg.svd(B)
    _ = None

    # trim them up
    UA = UA[:, :N]
    UB = UB[:, :Q]

    UA_UB_kron = np.kron(UA, UB)

    return UA_UB_kron


def full_kron_product_truncated_scipy_1d(x, N):
    # did it this way so that we don't explode memory constraints
    x_arr = np.arange(x) / (x - 1)
    N_arr = np.arange(N)
    A = np.zeros((x, N))
    for i, x_ in enumerate(x_arr):
        for j, n_ in enumerate(N_arr):
            A[i, j] = scipy.special.eval_chebyt(n_, x_)

    UA, _, _ = np.linalg.svd(A)

    # trim them up
    UA = UA[:, :N]

    return UA


def create_string(
    u,
    varname="test",
    eleorder=6,
    reduced=2,
    dim=3,
    print_to_2d=False,
    dimvarname="test",
):
    # if DTYPE == np.float64:
    #     outstr = "double "
    # else:
    #     raise NotImplementedError("Not yet implemented!")

    u = u.T

    outstr = ""
    # np.set_printoptions(precision=18)

    outstr += f"void set_chebyshev_mat_ele{eleorder}_out{reduced}_dim{dim}(){{\n"

    # make sure we deallocate just in case

    outstr += f"if ({varname}_dim{dim} != nullptr) {{delete[] {varname}_dim{dim};}}\n\n"

    # then we can set stuff up
    outstr += f"    {dimvarname}_dim{dim}_comp = {u.shape[0]};\n"
    outstr += f"    {dimvarname}_dim{dim}_decomp = {u.shape[1]};\n"
    outstr += f"    // {varname}[{u.shape[0]}][{u.shape[1]}] = {{\n"
    # set the variable to

    if print_to_2d:
        outstr += (
            f"    {varname}_dim{dim} = new double[{u.shape[0]}][{u.shape[1]}] {{\n"
        )
        for ii in range(u.shape[0]):
            outstr += "        {"
            counter = 0
            for jj in range(u.shape[1]):
                outstr += f"{u[ii, jj]}"
                if jj < u.shape[1] - 1:
                    outstr += ", "
                counter += 1
                if counter == 3:
                    counter = 0
                    outstr += "\n     "
            outstr += "}"
            if ii < u.shape[0] - 1:
                outstr += ","
            outstr += "\n"
        outstr += "    };\n"
    else:
        outstr += f"    {varname}_dim{dim} = new double[{u.shape[0] * u.shape[1]}] {{\n"

        for ii in range(u.shape[0]):
            outstr += "        "
            counter = 0
            for jj in range(u.shape[1]):
                outstr += f"{u[ii, jj]}, "
                counter += 1
                if counter == 3:
                    counter = 0
                    outstr += "\n        "
        outstr += "    };"

    outstr += "};\n\n"

    return outstr


if __name__ == "__main__":

    to_write = ""

    # we need to go through all of the different block sizes

    # calculate nx, ny, and nz
    n_pts = ELE_ORDER - 1
    # calculate N, Q, S

    for N in range(1, n_pts):
        print(N)
        u = full_kron_product_truncated_scipy(n_pts, n_pts, n_pts, N, N, N)
        to_write += create_string(
            u, varname="A_cheb", eleorder=ELE_ORDER, reduced=N, dim=3, dimvarname="cheb"
        )

        u = full_kron_product_truncated_scipy_2d(n_pts, n_pts, N, N)
        to_write += create_string(
            u, varname="A_cheb", eleorder=ELE_ORDER, reduced=N, dim=2, dimvarname="cheb"
        )

        u = full_kron_product_truncated_scipy_1d(n_pts, N)
        to_write += create_string(
            u, varname="A_cheb", eleorder=ELE_ORDER, reduced=N, dim=1, dimvarname="cheb"
        )

    with open(output_folder / f"cheb_transform_ele{ELE_ORDER}.inc.h", "w") as f:
        f.write(to_write)


# print(create_string(u, "test"))
