#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

int main(int argc, char** argv)
{
    int n = 1000000;
    int count =0;
    int arr[1000000];
    struct timespec startTime, endTime;
    unsigned long long elapsed_ns; double elapsed_s;
    
    #pragma omp parallel for
    for (int i=0; i<1000000; i++)
        arr[i] = 0;
    
    printf("number of devices %i\n",omp_get_num_devices()); //should be 1 for the GPU

    #pragma omp target data map(arr)
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    //   teams distribute   num_threads(16)
    #pragma omp target parallel for 
    for (int i=0; i<1000000; i++)
    {
        //printf("%i ",omp_get_thread_num());
        arr[i] = omp_get_thread_num();
    }
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    elapsed_ns = (endTime.tv_sec-startTime.tv_sec)*1000000000 + (endTime.tv_nsec-startTime.tv_nsec);
    elapsed_s = ((double)elapsed_ns)/1000000000.0;
    for (int i=0; i<1000000; i++){
        //printf("%i ", arr[i]);
    }
    printf("\ntime elapsed %f s or %lld ns\n",elapsed_s,elapsed_ns);
}