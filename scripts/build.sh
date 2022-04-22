#!/usr/bin/bash

module load CMake/3.18.4-GCCcore-10.2.0
module load gcc/8.2.0
module load OpenMPI/4.1.1-GCC-10.3.0

mkdir ../build
cd ../build
cmake ..
make

