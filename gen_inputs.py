#!/usr/bin/env python3
"""Ornek matris dosyalarini uretir (tekrarlanabilir, dosyadan okuma formati)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_matrix_txt(path: Path, mat: np.ndarray) -> None:
    n = mat.shape[0]
    lines = [str(n)]
    for i in range(n):
        lines.append(" ".join(str(x) for x in mat[i]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="a.txt / b.txt veya N boyutunda ornek uret")
    p.add_argument("--n", type=int, default=256, help="Matris boyutu (kare)")
    p.add_argument("--prefix", default="", help="Dosya: a{prefix}.txt (bos: a.txt)")
    args = p.parse_args()
    n = args.n
    rng = np.random.default_rng(4016)
    a = rng.integers(0, 9, size=(n, n), dtype=np.int64).astype(np.float64)
    b = rng.integers(0, 9, size=(n, n), dtype=np.int64).astype(np.float64)
    suf = str(args.prefix) if args.prefix else ""
    ap = Path(f"a{suf}.txt" if suf else "a.txt")
    bp = Path(f"b{suf}.txt" if suf else "b.txt")
    write_matrix_txt(ap, a)
    write_matrix_txt(bp, b)
    print(f"Yazildi: {ap}, {bp} (N={n})")


if __name__ == "__main__":
    main()
