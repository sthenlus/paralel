#!/usr/bin/env python3
"""
MPI ile C = A @ B — basit MPI_Scatter / MPI_Gather (mpi4py: scatter / gather).
Her surece esit satir: N % P == 0 olmalidir (ornek N=16, P=1,2,4,8,16; varsayilan a.txt / b.txt).
Kullanim: mpirun -np P python3 main.py [a.txt] [b.txt]
"""
from __future__ import annotations

import sys
import time

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
        if na != nb:
            print(f"Hata: A ve B boyutlari esit degil ({na} vs {nb}).")
            comm.Abort(1)
        n = na
        if n % size != 0:
            print(
                f"Hata: MPI_Scatter icin N={n} sayisi P={size} ile tam bolunmeli (N % P == 0)."
            )
            comm.Abort(1)
        start_time = time.perf_counter()
    else:
        a = None
        b = None

    n = comm.bcast(n, root=0)
    b = comm.bcast(b, root=0)

    rows_per_proc = n // size
    chunk = rows_per_proc * n
    chunks = (
        [a[i * chunk : (i + 1) * chunk] for i in range(size)] if rank == 0 else None
    )
    sub_a = comm.scatter(chunks, root=0)

    sub_c = [0.0] * chunk
    for i in range(rows_per_proc):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += sub_a[i * n + k] * b[k * n + j]
            sub_c[i * n + j] = s

    res = comm.gather(sub_c, root=0)

    if rank == 0:
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"Python - N={n}, Proc={size}, Sure={elapsed:.6f} s")


if __name__ == "__main__":
    main()
