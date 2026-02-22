CC = gcc
CFLAGS = -Wall -Wextra -std=c17 -I..

SRC = assignment/assignment.cu
OBJ = $(SRC:.c=.o)
TARGET = assignment

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

assignment/%.o: assignment/%.cu ../CodeTimer.h
	$(CC) $(CFLAGS) -c -o $@ $

clean:
	rm -f $(OBJ) $(TARGET)