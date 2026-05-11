CC = mpicc
CFLAGS = -O3 -Wall -std=c11

main: main.c
	$(CC) $(CFLAGS) -o main main.c

clean:
	rm -f main

.PHONY: clean
