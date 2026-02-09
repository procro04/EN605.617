//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// #define N 1000000
long N;

void init_vectors(int* v1, int* v2, int min, int max)
{
    for (int i = 0; i < N; i++) {
        v1[i] = min + rand() % (max - min + 1);
        v2[i] = min + rand() % (max - min + 1);
    }
}

void vector_calc(int* v1, int* v2, int* v3, int pattern)
{	
	int condition;
        
	for (int i = 0; i < N; ++i)
	{
		switch(pattern) {
			case 0: // Predictable - all true
				condition = 1;
				break;
			case 1: // Unpredictable - based on data
				condition = (v1[i] % 2 == 0);
				break;
			case 2: // Predictable alternating
				condition = (i % 2 == 0);
				break;
		}
		
		int result;
		// Equal work in both branches
		if (condition) {
			result = v1[i];
			for (int j = 0; j < 100; j++) {
				result = result * 3 + v2[i];
			}
		} else {
			result = v2[i];
			for (int j = 0; j < 100; j++) {
				result = result * 3 + v1[i];
			}
		}
		v3[i] = result;
	}
}

int main(int argc, char** argv)
{
	int min = -1000;
	int max = 1000;
	int pattern = 0;
	if (argc <= 2)
	{
		printf("Must specify the number of elements to compute.\n");
	}
	if (argc == 3) {
		pattern = strtol(argv[2], NULL, 10);
	}
	N = strtol(argv[1], NULL, 10);

	printf("Allocating vectors\n");
	// Seed the random number gen for populating the vectors
	srand(time(NULL));
	// Init vectors
	int* v1 = calloc(N , sizeof(int));
	printf("Allocated v1\n");
	int* v2 = calloc(N , sizeof(int));
	printf("Allocated v2\n");
	init_vectors(v1, v2, min, max);
	printf("Initialized both vectors\n");
	int* v3 = calloc(N , sizeof(int));

	struct timespec start, end;
    double time_spent;
    clock_gettime(CLOCK_MONOTONIC, &start);
	printf("Adding vectors\n");
	vector_calc(v1, v2, v3, pattern);
	printf("Vectors added\n");

	clock_gettime(CLOCK_MONOTONIC, &end);
    time_spent = (end.tv_sec - start.tv_sec) + 
                 (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Function took %f seconds to execute\n", time_spent);
	free(v1);
	free(v2);
}
