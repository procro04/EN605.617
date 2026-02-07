//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000

void init_vectors(int* v1, int* v2, int min, int max)
{
    for (int i = 0; i < N; i++) {
        v1[i] = min + rand() % (max - min + 1);
        v2[i] = min + rand() % (max - min + 1);
    }
}

void vector_add(int* v1, int* v2)
{	
	int v3[N];
	for (int i = 0; i < N; ++i)
	{
		v3[i] = v1[i] + v2[i];
	}
}

int main(int argc, char** argv)
{
	int min = -1000;
	int max = 1000;
	// Seed the random number gen for populating the vectors
	srand(time(NULL));
	// Init vectors
	int v1[N];
	int v2[N];
	init_vectors(v1, v2, min, max);

	struct timespec start, end;
    double time_spent;
    clock_gettime(CLOCK_MONOTONIC, &start);
	vector_add(v1, v2);

	clock_gettime(CLOCK_MONOTONIC, &end);
    time_spent = (end.tv_sec - start.tv_sec) + 
                 (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Function took %f seconds to execute\n", time_spent);
}
