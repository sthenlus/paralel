# BIL4016 — MPI ile Paralel Matris–Matris Çarpımı (C ve Python)

Kare matrisler **A** ve **B** için **C = A × B** hesabı; veriler `a.txt` / `b.txt` (N=4) ve `a256.txt` / `b256.txt` (N=256) dosyalarından okunur. **MPI**: dağıtım (Scatterv), **B** için yayın (Bcast), sonuç toplama (Gatherv).

## Donanım ve ortam (rapor için)

| Öğe | Değer |
|-----|--------|
| İşlemci | Intel Core i5, 10. nesil |
| Çekirdek | 8 (fiziksel) |
| Ortam | WSL (Linux), OpenMPI, `mpi4py` + NumPy (`.venv`) |

> PDF raporunda kendi cümlelerinle: P=16 senaryosu **çekirdek sayısından fazla MPI süreci** olduğu için *oversubscription*; P=8 sonrası hızlanmanın azalması beklenen davranıştır.

## Tanımlar (ödev 3. bölüm)

- **T_P(P)**: `mpirun -np P` ile ölçülen duvar saati süresi (program çıktısı).
- **Speedup**: \(S(P) = \dfrac{T_P(1)}{T_P(P)}\).
- **Verim**: \(E(P) = \dfrac{S(P)}{P}\).

Aşağıdaki **S** ve **E** değerleri bu tanımlarla tablolardan hesaplanmıştır (yuvarlanmış).

---

## Tablolar — N = 4 (`a.txt`, `b.txt`)

### C (MPI)

| P | T_P (s) | S | E |
|---:|---:|---:|---:|
| 1 | 0.000010 | 1.00 | 1.00 |
| 2 | 0.000013 | 0.77 | 0.39 |
| 4 | 0.000042 | 0.24 | 0.06 |
| 8 | 0.000132 | 0.08 | 0.01 |
| 16 | 0.000151 | 0.07 | 0.00 |

### Python (mpi4py)

| P | T_P (s) | S | E |
|---:|---:|---:|---:|
| 1 | 0.001385 | 1.00 | 1.00 |
| 2 | 0.001521 | 0.91 | 0.46 |
| 4 | 0.002114 | 0.66 | 0.16 |
| 8 | 0.005845 | 0.24 | 0.03 |
| 16 | 0.073157 | 0.02 | 0.00 |

**Kısa yorum (N=4):** Problem çok küçük; sürenin büyük kısmı **MPI kolektifleri** ve başlangıç maliyetindedir. Bu yüzden **S** ve **E** “ideal paralel matris çarpımı” gibi davranmaz; ölçeklenme analizi için **N=256** tabloları anlamlıdır.

---

## Tablolar — N = 256 (`a256.txt`, `b256.txt`)

### C (MPI)

| P | T_P (s) | S | E |
|---:|---:|---:|---:|
| 1 | 0.024394 | 1.00 | 1.00 |
| 2 | 0.012649 | 1.93 | 0.96 |
| 4 | 0.008076 | 3.02 | 0.75 |
| 8 | 0.007587 | 3.22 | 0.40 |
| 16 | 0.013551 | 1.80 | 0.11 |

### Python (mpi4py)

| P | T_P (s) | S | E |
|---:|---:|---:|---:|
| 1 | 0.089594 | 1.00 | 1.00 |
| 2 | 0.049173 | 1.82 | 0.91 |
| 4 | 0.044079 | 2.03 | 0.51 |
| 8 | 0.040649 | 2.20 | 0.28 |
| 16 | 0.153781 | 0.58 | 0.04 |

**Kısa yorum (N=256):**

- **C:** P=1→8 aralığında **T_P** azalır, **S** artar; P=16’da süre tekrar büyür (**8 çekirdek** + iletişim maliyeti).
- **Python:** Aynı aralıkta iyileşme var; mutlak süre C’den uzun (dil + mpi4py + çalışma zamanı). P=16’da **S < 1** (tek süreçten yavaş) — oversubscription ve iletişim baskını.

---

## Grafik önerisi (ödev 4. bölüm)

- Eksenler: **x = P** (1, 2, 4, 8, 16), **y = S(P)**.
- En az bir grafik: **N=256** için **C ve Python** aynı figürde (iki seri).
- İsteğe bağlı: **y = E(P)** ikinci grafik.

---

## Derleme ve çalıştırma (WSL)

```bash
sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make
sed -i 's/\r$//' run_benchmarks.sh run_all_examples.sh   # gerekiyorsa (CRLF)
chmod +x run_benchmarks.sh run_all_examples.sh
./run_all_examples.sh
```

Tek ölçüm:

```bash
mpirun --oversubscribe -np 4 ./main a256.txt b256.txt
mpirun --oversubscribe -np 4 .venv/bin/python3 main.py a256.txt b256.txt
```

256’lık matris dosyalarını yeniden üretmek:

```bash
python3 gen_inputs.py --n 256 --prefix 256
```

---

## Dosyalar

| Dosya | Açıklama |
|--------|----------|
| `main.c` | C + MPI |
| `main.py` | Python + mpi4py |
| `Makefile` | `mpicc -O3` ile `main` |
| `a.txt`, `b.txt` | N=4 örnek |
| `a256.txt`, `b256.txt` | N=256 örnek |
| `gen_inputs.py` | Matris txt üretimi |
| `run_benchmarks.sh` | P = 1,2,4,8,16 otomatik |
| `run_all_examples.sh` | N=4 ve N=256 sırayla |

---

## Teslim hatırlatması (ders notu)

Zip: `main.c`, `main.py`, örnek `a.txt` / `b.txt`, PDF rapor. Bu `README.md` dersin zorunluluğu değilse zip’e eklemek opsiyoneldir.
