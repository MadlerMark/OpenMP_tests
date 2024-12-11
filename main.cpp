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
    //float x[ARR_SZ];float y[ARR_SZ];float z[ARR_SZ];
    float* x = new float[ARR_SZ];float* y = new float[ARR_SZ];float* z = new float[ARR_SZ];
    float a = 2.718281828;
    struct timespec startTime, endTime;
    unsigned long long elapsed_ns_gpu; double elapsed_s_gpu;
    unsigned long long elapsed_ns; double elapsed_s;
    unsigned long long elapsed_ns_data_mv; double elapsed_s_data_mv;

    #pragma omp parallel for
    for (int i=0; i<ARR_SZ; i++){
        x[i] = i;
        y[i] = ARR_SZ - i;
    }
    
    printf("number of devices %i\n",omp_get_num_devices()); //should be 1 for the GPU

    //DATA MOVEMENT SECTION:
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    #pragma omp target enter data map(to: a,x[0:ARR_SZ],y[0:ARR_SZ],z[0:ARR_SZ])
    { }
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    elapsed_ns_data_mv = (endTime.tv_sec-startTime.tv_sec)*1000000000 + (endTime.tv_nsec-startTime.tv_nsec);
    elapsed_s_data_mv = ((double)elapsed_ns_data_mv)/1000000000.0;
    //END DATA MOVEMENT SECTION

    //BEGIN COMPUTATION SECTION
    clock_gettime(CLOCK_MONOTONIC, &startTime); //starts timing after the data movement
    #pragma omp target teams //map(from:z[0:ARR_SZ])
    #pragma omp distribute parallel for 
    for (int i = 0; i < ARR_SZ; i++) {
        z[i] = a * x[i] + y[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &endTime); //only return data movement is timed
    elapsed_ns_gpu = (endTime.tv_sec-startTime.tv_sec)*1000000000 + (endTime.tv_nsec-startTime.tv_nsec);
    elapsed_s_gpu = ((double)elapsed_ns_gpu)/1000000000.0;
    printf("finished GPU section, moving on to CPU...\n");
    //END COMPUTATION SECTION

    //RESET DATA FOR CPU FOR FAIRNESS
    #pragma omp parallel for
    for (int i=0; i<ARR_SZ; i++){
        x[i] = i;
        y[i] = ARR_SZ - i;
    }

    //BEGIN CPU SECTION
    clock_gettime(CLOCK_MONOTONIC, &startTime); //   teams distribute   num_threads(16) 
    saxpy_cpu(a,z,x,y,ARR_SZ);
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    elapsed_ns = (endTime.tv_sec-startTime.tv_sec)*1000000000 + (endTime.tv_nsec-startTime.tv_nsec);
    elapsed_s = ((double)elapsed_ns)/1000000000.0;
    //END CPU SECTION

    printf("\nGPU time elapsed %f s or %lld ns\n",elapsed_s_gpu,elapsed_ns_gpu);
    printf("CPU time elapsed %f s or %lld ns\n",elapsed_s,elapsed_ns);
    printf("Data movement time %f s or %lld ns\n",elapsed_s_data_mv,elapsed_ns_data_mv);
    delete [] x;
    delete [] y;
    delete [] z;
}