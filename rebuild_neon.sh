cd neon_py_bindings
rm -fr cmake-build-debug
mkdir cmake-build-debug
cd cmake-build-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DNEON_WARP_COMPILATION=ON ..
cmake --build . --target libNeonPy -j 30
