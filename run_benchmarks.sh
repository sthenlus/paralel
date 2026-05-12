#!/usr/bin/env bash
# WSL / Linux: mpicc + mpi4py kurulu olmali.
# Kullanim: ./run_benchmarks.sh [a.txt b.txt]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [[ -x "$DIR/.venv/bin/python3" ]]; then
  PYTHON="$DIR/.venv/bin/python3"
else
  PYTHON="python3"
fi

A="${1:-a.txt}"
B="${2:-b.txt}"
LABEL="${A}_${B}"
echo "=== Ornek: $A + $B ==="

# MPI_Scatter: N % P == 0. N=16 ve N=256 icin P=1,2,4,8,16 uyumludur.
NPS=(1 2 4 8 16)

if [[ ! -x ./main ]]; then
  echo "Derleniyor: make"
  make
fi

for np in "${NPS[@]}"; do
  echo "--- C, P=$np ---"
  mpirun --oversubscribe -np "$np" ./main "$A" "$B" || true
done

for np in "${NPS[@]}"; do
  echo "--- Python, P=$np ---"
  mpirun --oversubscribe -np "$np" "$PYTHON" main.py "$A" "$B" || true
done

echo "Bitti: $LABEL"
