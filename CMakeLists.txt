cmake_minimum_required(VERSION 3.10)

option(DEBUG "If ON, executable will try to diagnose where failures occur." OFF)

# set compilers
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)

# set the project name
project(HybridSBP VERSION 2022.02.04)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(extern/hungarian)
add_subdirectory(extern/abseil-cpp)
add_subdirectory(extern/robin-map)

set(INCLUDE_DIRS
        include
        include/blockmodel
        include/blockmodel/sparse
        extern/tclap-1.2.2/include
        patches
        extern/hungarian
        extern/abseil-cpp
        extern/robin-map/include
        )

# add the executable
add_library(SBP STATIC
        src/graph.cpp
        src/utils.cpp
        src/sbp.cpp
        src/entropy.cpp
        src/evaluate.cpp
        src/common.cpp
        src/finetune.cpp
        src/block_merge.cpp
        src/partition.cpp
        src/blockmodel/blockmodel.cpp
        src/blockmodel/blockmodel_triplet.cpp
        src/blockmodel/dist_blockmodel.cpp
        src/blockmodel/sparse/dict_matrix.cpp
        src/blockmodel/sparse/dict_transpose_matrix.cpp
        #    blockmodel/sparse/dist_dict_matrix.cpp
        )
target_include_directories(SBP PUBLIC ${INCLUDE_DIRS})

add_executable(HybridSBP src/main.cpp)
target_include_directories(HybridSBP PUBLIC ${INCLUDE_DIRS})

add_executable(Influence src/influence.cpp)
target_include_directories(Influence PUBLIC ${INCLUDE_DIRS})

if (DEBUG)
    #    To check for heap overflows and such, use -fsanitize=address
    #    To run with valgrind, do NOT use -fsanitize=address
    target_compile_options(SBP PUBLIC -Wall -no-pie -fopenmp -g3 -fsanitize=address)
    target_link_libraries(SBP PUBLIC hungarian absl::flat_hash_map -fopenmp stdc++fs -g3 -fsanitize=address)
    target_link_libraries(HybridSBP PUBLIC SBP absl::flat_hash_map -g3 -fsanitize=address)
    target_link_libraries(Influence PUBLIC SBP absl::flat_hash_map -g3 -fsanitize=address)
    #    target_compile_options(SBP PUBLIC -Wall -no-pie -fopenmp -g3)
    #    target_link_libraries(SBP PUBLIC hungarian -fopenmp stdc++fs -g3)
    #    target_link_libraries(HybridSBP PUBLIC SBP -g3)
    #    target_link_libraries(Influence PUBLIC SBP -g3)
else (DEBUG)
    #    target_compile_options(HybridSBP PUBLIC -Wall -O3 -no-pie -fopenmp)
    #    target_link_libraries(HybridSBP PUBLIC SBP hungarian -fopenmp stdc++fs)
    target_compile_options(SBP PUBLIC -Wall -O3 -no-pie -fopenmp)
    target_link_libraries(SBP PUBLIC hungarian absl::hash absl::flat_hash_map -fopenmp stdc++fs)
    target_link_libraries(HybridSBP PUBLIC SBP absl::hash absl::flat_hash_map)
    target_link_libraries(Influence PUBLIC SBP absl::hash absl::flat_hash_map)
endif (DEBUG)

# Google Test Suite
enable_testing()
add_subdirectory(extern/googletest)
add_executable(Test
        test/blockmodel_test.cpp
        test/block_merge_test.cpp
        test/common_test.cpp
        test/dict_matrix_test.cpp
        test/dict_transpose_matrix_test.cpp
        test/entropy_test.cpp
        test/evaluate_test.cpp
        test/finetune_test.cpp
        test/toy_example.hpp)
target_include_directories(Test PUBLIC ${INCLUDE_DIRS})
include(GoogleTest)
target_link_libraries(Test SBP gtest_main hungarian absl::hash absl::flat_hash_map -fopenmp stdc++fs)
gtest_discover_tests(Test)
