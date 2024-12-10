#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv)
{
    #pragma omp target
    {
        printf("hello world from the GPU\n");
    }
    int n = 1000000;
    int count =0;
    #pragma omp target teams distribute parallel for
    for (int i=0; i<n; i++)
    {
        printf("%i ",omp_get_thread_num());
        count++;
    }
    printf("this is the count: %i\n", count);
}