CC=g++
CFLAGS=-c -Wall -O2 -fopenmp
LDFLAGS=-lgomp
SOURCES=main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=xrff-console

all: $(SOURCES) $(EXECUTABLE)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@ -I../Common

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) ../Calculation/*.o -o $@

clean:
	rm -rf *.o *~ $(EXECUTABLE)
