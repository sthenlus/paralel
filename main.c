/*
 * MPI parallel C = A * B — MPI_Scatter + MPI_Bcast + MPI_Gather (esit satir).
 * Kosul: N % P == 0 (ornek N=16, P=1,2,4,8,16; varsayilan a.txt / b.txt).
 * Usage: mpirun -np P ./main [a.txt] [b.txt]
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_matrix_file(const char *path, int *out_n, double **out_data) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Hata: '%s' acilamadi.\n", path);
        return -1;
    }
    int n;
    if (fscanf(f, "%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "Hata: '%s' ilk satirinda gecersiz N.\n", path);
        fclose(f);
        return -1;
    }
    size_t total = (size_t)n * (size_t)n;
    double *buf = (double *)malloc(total * sizeof(double));
    if (!buf) {
        fprintf(stderr, "Hata: '%s' icin bellek yetersiz (N=%d).\n", path, n);
        fclose(f);
        return -1;
    }
    for (size_t i = 0; i < total; i++) {
        if (fscanf(f, "%lf", &buf[i]) != 1) {
            fprintf(stderr, "Hata: '%s' matris elemani eksik.\n", path);
            free(buf);
            fclose(f);
            return -1;
        }
    }
    fclose(f);
    *out_n = n;
    *out_data = buf;
    return 0;
}

static void local_matmul_rows(const double *local_a, const double *b, double *local_c,
                              int local_rows, int n) {
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += local_a[(size_t)i * (size_t)n + (size_t)k] * b[(size_t)k * (size_t)n + (size_t)j];
            local_c[(size_t)i * (size_t)n + (size_t)j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *path_a = (argc > 1) ? argv[1] : "a.txt";
    const char *path_b = (argc > 2) ? argv[2] : "b.txt";

    int n = 0;
    double *a_full = NULL;
    double *b_full = NULL;

    if (rank == 0) {
        if (read_matrix_file(path_a, &n, &a_full) != 0)
            MPI_Abort(MPI_COMM_WORLD, 1);
        int nb;
        double *b_tmp = NULL;
        if (read_matrix_file(path_b, &nb, &b_tmp) != 0) {
            free(a_full);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (nb != n) {
            fprintf(stderr, "Hata: A ve B boyutlari esit degil (%d vs %d).\n", n, nb);
            free(a_full);
            free(b_tmp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        b_full = b_tmp;
        if (n % size != 0) {
            fprintf(stderr,
                    "Hata: MPI_Scatter icin N=%d, P=%d — N %% P == 0 olmali.\n", n, size);
            free(a_full);
            free(b_full);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_rows = n / size;
    int chunk = local_rows * n;
    double *local_a = (double *)malloc((size_t)chunk * sizeof(double));
    double *local_c = (double *)malloc((size_t)chunk * sizeof(double));
    double *b_local = (double *)malloc((size_t)n * (size_t)n * sizeof(double));

    if (!local_a || !local_c || !b_local) {
        if (rank == 0) {
            free(a_full);
            free(b_full);
        }
        free(local_a);
        free(local_c);
        free(b_local);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Scatter(a_full, chunk, MPI_DOUBLE, local_a, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
        memcpy(b_local, b_full, (size_t)n * (size_t)n * sizeof(double));
    MPI_Bcast(b_local, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_matmul_rows(local_a, b_local, local_c, local_rows, n);

    double *c_full = NULL;
    if (rank == 0)
        c_full = (double *)malloc((size_t)n * (size_t)n * sizeof(double));

    MPI_Gather(local_c, chunk, MPI_DOUBLE, c_full, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("C MPI (C): N=%d, P=%d, T_P=%.6f s\n", n, size, t1 - t0);
        fflush(stdout);
        free(a_full);
        free(b_full);
        free(c_full);
    }

    free(local_a);
    free(local_c);
    free(b_local);

    MPI_Finalize();
    return 0;
}
