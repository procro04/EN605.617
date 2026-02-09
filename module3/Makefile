all: c-assignment cu-assignment

c-assignment: assignment.c
	gcc assignment.c -o c-assignment

cu-assignment: assignment.cu
	nvcc assignment.cu -o cu-assignment

clean:
	rm -f c-assignment cu-assignment