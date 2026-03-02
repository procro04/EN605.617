
# Module 6
Folder for module 6 assignment and test code.

## Assignment
- An `assignment.cu` file.
- There is also a `CodeTimer.h` header file that I import from the main directory
in order to promote easy timing of the cuda functions.

### Building
1. First cd to the proper location and make a build directory.
```bash
cd module6/
mkdir build; cd build
```
2. Next run cmake. Note since I was running on WSL, I had to specify my architecture.
Feel free to modify the `CMakeLists.txt` file to your specific architecture. To find
your specific architecture, run the nvidia command below. You can also pass the flag
`-DCMAKE_CUDA_ARCHITECTURES=XX` to pass your arch.
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```
3. Now actually run the cmake commands. Note you can pass flags to compile with
or without debug or any other flags supported by cmake.
```bash
cmake ..
cmake --build .
```

### Running
- `./assignment` runs all memory types with the standard input.
```bash
./assignment <total threads> <block size> <num elements> <pattern> <num streams>
```
- Pattern is to test branching that was carried over from the module 3 kernel code.
