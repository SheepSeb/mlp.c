CC=gcc
CFLAGS=-lm -O3


run: clean main
	./main


clean:
	rm -f main

main: main.c
	$(CC) -o main main.c $(CFLAGS)

.PHONY: run clean
