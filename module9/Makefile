CXX     = nvcc
CXXFLAGS = -std=c++17
LIBS     = -lcurand -lcufft
 
signal_fft: signal_fft.cu
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@
 
clean:
	rm -f signal_fft
