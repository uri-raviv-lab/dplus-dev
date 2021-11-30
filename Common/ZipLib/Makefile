# Final names of binaries
EXECUTABLE = Bin/zipsample
SO_LIBRARY = Bin/libzip.a

# C & C++ compiler
CC       = gcc
CXX      = g++
CFLAGS    = -Wno-enum-conversion -O3
CXXFLAGS  = -std=c++11 -O3

# Linker flags
LDFLAGS   = -pthread

# Sources of external libraries
SRC_ZLIB  = $(wildcard Source/extlibs/zlib/*.c)

# ZipLib sources
SRC = \
		$(wildcard Source/*.cpp)        \
		$(wildcard Source/detail/*.cpp)

# Object files			
OBJS = \
		$(SRC:.cpp=.o)	   \
		$(SRC_ZLIB:.c=.o)  

# Rules
all: $(SO_LIBRARY)

$(EXECUTABLE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) Source/Sample/Main.cpp -o $@ $^

$(SO_LIBRARY): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf `find Source -name '*.o'` ziplib.tar.gz Bin/*.zip Bin/out* $(SO_LIBRARY)

tarball:
	tar -zcvf ziplib.tar.gz *
	