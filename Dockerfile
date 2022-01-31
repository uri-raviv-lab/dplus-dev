# This Dockerfile builds the D+ backend libraries, and the Python API

# Step 1 - build the backend libraries
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 AS cplusplus

# Step 1.1 - Install the necessary Ubuntu Packages
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt install -y cmake libboost-all-dev

# Step 1.2 - Copy the source files we need
COPY Backend /src/Backend
COPY BackendCommunication /src/BackendCommunication
COPY Common /src/Common
COPY Conversions /src/Conversions
COPY DebyeCalculator /src/DebyeCalculator
COPY pdbgen /src/pdbgen
COPY PDBManipulator /src/PDBManipulator
COPY PDBReaderLib /src/PDBReaderLib
COPY PDBUnits /src/PDBUnits
COPY Population /src/Population
COPY PropertySheets /src/PropertySheets
COPY backend_version.h /src
COPY CMakeLists.txt /src

# Step 1.3 - Build
RUN mkdir /src/build
WORKDIR /src/build
RUN cmake ..
RUN make

# Step 1.4 - Get files ready for the next stip
RUN mkdir /output
RUN cp /src/build/Backend/lib* /output
RUN cp /src/build/pdbgen /output
RUN cp /src/build/debye /output

# Step 2 - build the Python API
FROM quay.io/pypa/manylinux_2_24_x86_64 AS python
WORKDIR /src/PythonInterface

COPY --from=cplusplus /output /src/PythonInterface/lib 
COPY ./PythonInterface/requirements.txt /src/PythonInterface/requirements.txt
# Install all requirements for all Python versions
RUN /opt/python/cp37-cp37m/bin/pip install -r requirements.txt
RUN /opt/python/cp38-cp38/bin/pip install -r requirements.txt
RUN /opt/python/cp39-cp39/bin/pip install -r requirements.txt
RUN /opt/python/cp310-cp310/bin/pip install -r requirements.txt

# Now we can copy everything else - first the none-dplus-api files
COPY ./Backend /src/Backend
COPY ./Common /src/Common
COPY ./Conversions /src/Conversions

# And finally, the dplus-api source files.
COPY ./PythonInterface /src/PythonInterface

# Prepare for building the wheel - precompile pyx files and move header files around
RUN /opt/python/cp38-cp38/bin/python setup.py prepare

# Now we can build all the wheels
RUN /opt/python/cp38-cp38/bin/pip wheel .
RUN /opt/python/cp37-cp37m/bin/pip wheel .
RUN /opt/python/cp39-cp39/bin/pip wheel .
RUN /opt/python/cp310-cp310/bin/pip wheel .

# RUN mkdir /wheels
# RUN find . -name "dplus_api*whl" -exec auditwheel repair {} -w /wheels \;

# Have an entrypoint that allows further work on the container, until this Docker file is complete
ENTRYPOINT [ "/bin/bash" ]