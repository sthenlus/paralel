#!/usr/bin/env bash
# WSL / Linux: mpicc + mpi4py kurulu olmali.
# Kullanim: ./run_benchmarks.sh [a.txt b.txt]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# PEP 668: projede .venv kullan (bir kez: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt)
if [[ -x "$DIR/.venv/bin/python3" ]]; then
  PYTHON="$DIR/.venv/bin/python3"
else
  PYTHON="python3"
fi

A="${1:-a.txt}"
B="${2:-b.txt}"
LABEL="${A}_${B}"
echo "=== Ornek: $A + $B ==="

if [[ ! -x ./main ]]; then
  echo "Derleniyor: make"
  make
fi

NPS=(1 2 4 8 16)
for np in "${NPS[@]}"; do
  echo "--- C, P=$np ---"
  mpirun --oversubscribe -np "$np" ./main "$A" "$B" || true
done

# Python/NumPy: her MPI surecinde tek BLAS thread (main.py icinde de ayarli)
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

for np in "${NPS[@]}"; do
  echo "--- Python, P=$np ---"
  mpirun --oversubscribe -np "$np" "$PYTHON" main.py "$A" "$B" || true
done

echo "Bitti: $LABEL"
