# Building and Running
```bash
cd module3
make
```
## assignment.c
```bash
./c-assignment <N array elements> <Branch Type 0-2>
# Example
./c-assignment 10000000 0
```

## assignment.cu
```bash
./cu-assignment <thread count> <block size> <N array elements> <Branch type 0-2>
# Example 
./cu-assignment 512 64 100000000 0
```

# Notes
- All memory is dynamic, you will only be limited by your RAM and GPU compute sizes.
- There is information on the types of branches inside of the `vector_calc` functions
