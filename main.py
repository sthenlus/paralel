#!/usr/bin/env python3
"""
MPI ile C = A @ B — MPI_Scatterv / MPI_Gatherv (mpi4py: scatterv / gatherv).
Her surece degisken satir sayisi: N % P == 0 olmak zorunda degil!
Kullanim: mpirun -np P python3 main.py [a.txt] [b.txt]
"""
from __future__ import annotations

import sys
import time

import numpy as np
from mpi4py import MPI


def read_matrix(filename: str) -> tuple[int, list[float]]:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            raise ValueError("bos dosya")
        n = int(lines[0].strip())
        matrix: list[float] = []
        for line in lines[1:]:
            matrix.extend(float(x) for x in line.split())
        if len(matrix) != n * n:
            raise ValueError(
                f"{filename}: beklenen {n * n} eleman, okunan {len(matrix)} (N={n})"
            )
        return n, matrix
    except FileNotFoundError:
        print(f"Hata: {filename} dosyasi bulunamadi!")
        MPI.COMM_WORLD.Abort(1)
    except (ValueError, IndexError) as e:
        print(f"Hata: {filename} okunamadi: {e}")
        MPI.COMM_WORLD.Abort(1)


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    path_a = sys.argv[1] if len(sys.argv) > 1 else "a.txt"
    path_b = sys.argv[2] if len(sys.argv) > 2 else "b.txt"

    n = None
    a = None
    b = None

    if rank == 0:
        na, a = read_matrix(path_a)
        nb, b = read_matrix(path_b)
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        if na != nb:
            print(f"Hata: A ve B boyutlari esit degil ({na} vs {nb}).")
            comm.Abort(1)
        n = na
        start_time = time.perf_counter()
    else:
        a = None
        b = None

    n = comm.bcast(n, root=0)
    b = comm.bcast(b, root=0)

    # Scatterv icin: her surece degisken satir sayisi ver
    if rank == 0:
        base_rows = n // size
        extra_rows = n % size
        sendcounts = []
        displacements = []
        disp = 0
        for p in range(size):
            rows = base_rows + (1 if p < extra_rows else 0)
            count = rows * n
            sendcounts.append(count)
            displacements.append(disp)
            disp += count
    else:
        sendcounts = None
        displacements = None
    
    sendcounts = comm.bcast(sendcounts, root=0)
    
    local_chunk = sendcounts[rank]
    sub_a = np.empty(local_chunk, dtype=np.float64)
    
    # Scatterv - NumPy array ile
    comm.Scatterv([a, (sendcounts, displacements)], sub_a, root=0)

    rows_per_proc = local_chunk // n
    sub_c = np.zeros(local_chunk, dtype=np.float64)
    for i in range(rows_per_proc):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += sub_a[i * n + k] * b[k * n + j]
            sub_c[i * n + j] = s

    # Gatherv - NumPy array ile
    c_full = None
    if rank == 0:
        c_full = np.empty(n * n, dtype=np.float64)
    
    comm.Gatherv(sub_c, [c_full, (sendcounts, displacements)], root=0)

    if rank == 0:
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"Python - N={n}, Proc={size}, Sure={elapsed:.6f} s")


if __name__ == "__main__":
    main()
