cmake_minimum_required(VERSION 3.8)
project(JSONiq-MLIR)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-fno-rtti)

# Set up LLVM
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Set up MLIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

include_directories(${MLIR_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH
    ${LLVM_CMAKE_DIR}
    ${MLIR_CMAKE_DIR}
)

include(AddLLVM)
include(TableGen)
include(AddMLIR)

# Set up tablegen targets
set(LLVM_TARGET_DEFINITIONS src/Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls "-I${CMAKE_CURRENT_SOURCE_DIR}/src")
mlir_tablegen(Ops.cpp.inc -gen-op-defs "-I${CMAKE_CURRENT_SOURCE_DIR}/src")
add_public_tablegen_target(JSONiqOpsIncGen)

set(LLVM_TARGET_DEFINITIONS src/Combine.td)
mlir_tablegen(Combine.inc -gen-rewriters "-I${CMAKE_CURRENT_SOURCE_DIR}/src")
add_public_tablegen_target(JSONiqCombineIncGen)

# Set up main executable
add_executable(jsoniqc
        src/Combine.cpp
        src/dialect.cpp
        src/main.cpp
    )
llvm_update_compile_flags(jsoniqc)
add_dependencies(jsoniqc JSONiqOpsIncGen JSONiqCombineIncGen)

# Link against LLVM and MLIR libraries
llvm_map_components_to_libnames(LLVM_LIBS support core irreader)
target_link_libraries(jsoniqc
    PUBLIC
        ${LLVM_LIBS}
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTransforms
    )
