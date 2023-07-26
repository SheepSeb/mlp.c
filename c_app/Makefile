CC=gcc
CFLAGS=-lm -O3

run: main
	./main

main: main.c
	$(CC) -o main main.c $(CFLAGS)

clean:
	rm -f main

.PHONY: run clean
