#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

#define ARR_SZ 100000000
#define NUM_BLOCKS 100

// void saxpy(float a,float* z, float* x, float* y, int sz) {
//     #pragma omp target \
//     map(to:a, x, y) map(from:z)
//     #pragma omp teams default(shared)
//     #pragma omp distribute parallel for 
//     for (int i = 0; i < sz; i++) {
//         z[i] = a * x[i] + y[i];
//     }
// }

void saxpy_cpu(float a,float* z, float* x, float* y, int sz){
    #pragma omp parallel for
    for (int i = 0; i < sz; i++) {
        z[i] = a * x[i] + y[i];
    }
}

int main(int argc, char** argv)
{
    struct timespec startTime_tot, endTime_tot;
    float a = 2.718281828;
    struct timespec startTime, endTime;
    struct timespec startTime2,endTime2;
    struct timespec startTime3,endTime3;
    unsigned long long elapsed_ns_gpu; double elapsed_s_gpu;
    unsigned long long elapsed_ns; double elapsed_s;
    unsigned long long elapsed_ns_data_mv; double elapsed_s_data_mv;
    float* x = new float[ARR_SZ];float* y = new float[ARR_SZ];float* z = new float[ARR_SZ];
    clock_gettime(CLOCK_MONOTONIC,&startTime_tot);
    #pragma omp parallel for
    for (int i=0; i<ARR_SZ; i++){
        x[i] = i; y[i] = ARR_SZ - i;
    }
    //DATA MOVEMENT SECTION:
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    #pragma omp target enter data map(to: a,x[0:ARR_SZ],y[0:ARR_SZ],z[0:ARR_SZ])
    { }
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    //END DATA MOVEMENT SECTION
    //printf("number of devices %i\n",omp_get_num_devices()); //should be 1 for the GPU
    //BEGIN COMPUTATION SECTION
    clock_gettime(CLOCK_MONOTONIC, &startTime2); //starts timing after the data movement
    #pragma omp target teams //map(from:z[0:ARR_SZ])
    #pragma omp distribute parallel for 
    for (int i = 0; i < ARR_SZ; i++) {
        z[i] = a * x[i] + y[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &endTime2); //only return data movement is timed
    //printf("finished GPU section, moving on to CPU...\n");
    //END COMPUTATION SECTION
    //RESET DATA FOR CPU FOR FAIRNESS
    #pragma omp parallel for
    for (int i=0; i<ARR_SZ; i++){
        x[i] = i;
        y[i] = ARR_SZ - i;
    }
    //BEGIN CPU SECTION
    clock_gettime(CLOCK_MONOTONIC, &startTime3); //   teams distribute   num_threads(16) 
    saxpy_cpu(a,z,x,y,ARR_SZ);
    clock_gettime(CLOCK_MONOTONIC, &endTime3);
    //END CPU SECTION
    clock_gettime(CLOCK_MONOTONIC,&endTime_tot);
    elapsed_ns_data_mv = (endTime.tv_sec-startTime.tv_sec)*1000000000 + (endTime.tv_nsec-startTime.tv_nsec);
    elapsed_s_data_mv = ((double)elapsed_ns_data_mv)/1000000000.0;
    elapsed_ns_gpu = (endTime2.tv_sec-startTime2.tv_sec)*1000000000 + (endTime2.tv_nsec-startTime2.tv_nsec);
    elapsed_s_gpu = ((double)elapsed_ns_gpu)/1000000000.0;
    elapsed_ns = (endTime3.tv_sec-startTime3.tv_sec)*1000000000 + (endTime3.tv_nsec-startTime3.tv_nsec);
    elapsed_s = ((double)elapsed_ns)/1000000000.0;
    printf("\nGPU time elapsed %f s or %lld ns\n",elapsed_s_gpu,elapsed_ns_gpu);
    printf("CPU time elapsed %f s or %lld ns\n",elapsed_s,elapsed_ns);
    printf("Data movement time %f s or %lld ns\n",elapsed_s_data_mv,elapsed_ns_data_mv);
    delete [] x;
    delete [] y;
    delete [] z;
    
    elapsed_ns = (endTime_tot.tv_sec-startTime_tot.tv_sec)*1000000000 + (endTime_tot.tv_nsec-startTime_tot.tv_nsec);
    elapsed_s = ((double)elapsed_ns)/1000000000.0;
    printf("total program time elapsed %f s or %lld ns\n",elapsed_s,elapsed_ns);
}