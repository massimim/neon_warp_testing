#!/bin/bash

CUDA_PATH=/usr/local/cuda

SCRIPT_DIR=$(dirname $(realpath ${BASH_SOURCE}))

WARP_DIR=$(realpath "${SCRIPT_DIR}/../../warp")
NEON_DIR=$(realpath "${SCRIPT_DIR}/../../neon_py_bindings")

echo "Warp dir: $WARP_DIR"
echo "Neon dir: $NEON_DIR"

if [ ! -e "${CUDA_PATH}" ]; then
    echo "*** Failed to find CUDA in ${CUDA_PATH}, please update CUDA_PATH in the script"
    exit 1
fi

cmd="${CUDA_PATH}/bin/nvcc"
# compiler options
cmd="${cmd} --std=c++17"
cmd="${cmd} --compiler-options -shared,-fPIC"
cmd="${cmd} -DNEON_COMPILER_GCC"
cmd="${cmd} -I${WARP_DIR}"
cmd="${cmd} -I${NEON_DIR}/libNeonCore/include"
cmd="${cmd} -I${NEON_DIR}/libNeonDomain/include"
cmd="${cmd} -I${NEON_DIR}/libNeonSet/include"
cmd="${cmd} -I${NEON_DIR}/libNeonSys/include"
cmd="${cmd} -I${NEON_DIR}/cmake-build-debug/libNeonCore"
cmd="${cmd} -I${NEON_DIR}/cmake-build-debug/libNeonSys"
cmd="${cmd} -I${NEON_DIR}/cmake-build-debug/_deps/spdlog-src/include"
cmd="${cmd} -I${NEON_DIR}/cmake-build-debug/_deps/rapidjson-src/include"
# linker options
cmd="${cmd} --linker-options=--no-undefined"
cmd="${cmd} -L${NEON_DIR}/cmake-build-debug/libNeonCore"
cmd="${cmd} -L${NEON_DIR}/cmake-build-debug/libNeonDomain"
cmd="${cmd} -L${NEON_DIR}/cmake-build-debug/libNeonSet"
cmd="${cmd} -L${NEON_DIR}/cmake-build-debug/libNeonSys"
cmd="${cmd} -llibNeonCore"
cmd="${cmd} -llibNeonDomain"
cmd="${cmd} -llibNeonSet"
cmd="${cmd} -llibNeonSys"
# input/output
cmd="${cmd} ${SCRIPT_DIR}/test_native_launch.cpp"
cmd="${cmd} -o ${SCRIPT_DIR}/test_native_launch.so"

# build it
echo ${cmd}
eval "${cmd}"

# create symbolic links for dependencies in origin directory
if [ ! -e "${SCRIPT_DIR}/liblibNeonCore.so" ]; then
    ln -s "${NEON_DIR}/cmake-build-debug/libNeonCore/liblibNeonCore.so" "${SCRIPT_DIR}/liblibNeonCore.so"
fi
if [ ! -e "${SCRIPT_DIR}/liblibNeonDomain.so" ]; then
    ln -s "${NEON_DIR}/cmake-build-debug/libNeonDomain/liblibNeonDomain.so" "${SCRIPT_DIR}/liblibNeonDomain.so"
fi
if [ ! -e "${SCRIPT_DIR}/liblibNeonSet.so" ]; then
    ln -s "${NEON_DIR}/cmake-build-debug/libNeonSet/liblibNeonSet.so" "${SCRIPT_DIR}/liblibNeonSet.so"
fi
if [ ! -e "${SCRIPT_DIR}/liblibNeonSys.so" ]; then
    ln -s "${NEON_DIR}/cmake-build-debug/libNeonSys/liblibNeonSys.so" "${SCRIPT_DIR}/liblibNeonSys.so"
fi

# update RPATH to load dependencies from origin directory
patchelf --set-rpath '$ORIGIN' "${SCRIPT_DIR}/test_native_launch.so"
