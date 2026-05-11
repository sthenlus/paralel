#!/usr/bin/env bash
# N=4 ve N=256 ornekleri.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
chmod +x run_benchmarks.sh 2>/dev/null || true
echo "######## N=4 (a.txt, b.txt) ########"
./run_benchmarks.sh a.txt b.txt
echo ""
echo "######## N=256 (a256.txt, b256.txt) ########"
./run_benchmarks.sh a256.txt b256.txt
