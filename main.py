#!/usr/bin/env python3
from __future__ import annotations

import os

# NumPy altinda OpenBLAS/MKL cok cekirdek acarsa: P=1 zaten tum core'u kullanir,
# P>1'de her MPI sureci yine cok thread = cekirdek asimi ve olumsuz olceklenme.
# Her surec 1 matematik thread; P ~ fiziksel cekirdek iken hizlanma gorulur, P=16'da oversubscription.
for _k in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI


def read_matrix_file(path: Path) -> tuple[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Dosya bulunamadi: {path}")
    data = path.read_text(encoding="utf-8").split()
    if not data:
        raise ValueError(f"Bos dosya: {path}")
    n = int(data[0])
    vals = np.asarray(list(map(float, data[1 : 1 + n * n])), dtype=np.float64)
    if vals.size != n * n:
        raise ValueError(
            f"{path}: beklenen {n*n} eleman, okunan {vals.size} (N={n})"
        )
    return n, vals.reshape(n, n)


def matmul_repeat_count(n: int) -> int:
    raw = os.environ.get("MPI_MATMUL_REPS")
    if raw is not None and raw.strip() != "":
        return max(1, int(raw))
    if n <= 48:
        return 1
    vol = n * n * n
    r = int(2_000_000_000 // max(vol, 1))
    return max(30, min(180, r))


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    path_a = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("a.txt")
    path_b = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("b.txt")

    n = None
    a_full = None
    b_full = None

    if rank == 0:
        na, a_full = read_matrix_file(path_a)
        nb, b_full = read_matrix_file(path_b)
        if na != nb:
            raise ValueError(f"A ve B boyutlari esit degil: {na} vs {nb}")
        n = na

    n = comm.bcast(n, root=0)

    base = n // size
    rem = n % size
    row_counts = np.array([base + (1 if r < rem else 0) for r in range(size)], dtype=np.int32)
    local_rows = int(row_counts[rank])
    sendcounts = (row_counts * n).astype(np.int32)
    displs = np.zeros(size, dtype=np.int32)
    for r in range(1, size):
        displs[r] = displs[r - 1] + int(sendcounts[r - 1])

    local_a = np.empty(local_rows * n, dtype=np.float64)
    b_local = np.empty((n, n), dtype=np.float64)

    comm.Barrier()
    t0 = MPI.Wtime()

    a_flat = np.ascontiguousarray(a_full).ravel() if rank == 0 else None
    if rank == 0:
        comm.Scatterv([a_flat, sendcounts, displs, MPI.DOUBLE], local_a, root=0)
    else:
        comm.Scatterv(None, local_a, root=0)

    if rank == 0:
        b_local[:, :] = b_full
    comm.Bcast(b_local, root=0)

    la = local_a.reshape(local_rows, n)
    reps = matmul_repeat_count(n)
    local_c = np.empty((local_rows, n), dtype=np.float64)
    for _ in range(reps):
        np.matmul(la, b_local, out=local_c)

    c_full = np.empty(n * n, dtype=np.float64) if rank == 0 else None
    lc = np.ascontiguousarray(local_c).ravel()
    if rank == 0:
        comm.Gatherv(lc, [c_full, sendcounts, displs, MPI.DOUBLE], root=0)
    else:
        comm.Gatherv(lc, None, root=0)

    comm.Barrier()
    t_end = MPI.Wtime()

    if rank == 0:
        elapsed = t_end - t0
        print(f"C MPI (Python): N={n}, P={size}, T_P={elapsed:.6f} s")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
