all: test

test: main.cpp
	gcc main.cpp -O3 -lstdc++ -fopenmp -foffload=nvptx-none -fno-stack-protector -fcf-protection=none -o test.o

.PHONY : clean
clean: 
	rm test.o
