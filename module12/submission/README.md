# Simple OpenCL Matrix Multiplier
## Building
```
mkdir build
cd build
cmake ..
cmake --build .
```
## Running
```
./matmul --m 4 --k 3 --n 5          # A(4x3) * B(3x5) = C(4x5)
./matmul --m 8 --k 8 --n 8 --verify # 8x8 multiply with CPU check
./matmul --m 512 --k 256 --n 512 --platform 1   # Specify a different platform
```