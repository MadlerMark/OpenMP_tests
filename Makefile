all: test

test: main.cpp
	gcc main.cpp -Og -fopenmp -foffload=nvptx-none -v -o test.o

.PHONY : clean
clean: 
	rm test.o
