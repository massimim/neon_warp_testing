git clone git@github.com:Autodesk/Neon.git -b py neon_py_bindings
git clone git@github.com:nvlukasz/warp.git -b external-source-support warp

cd neon_py_bindings
mkdir cmake-build-debug
cd cmake-build-debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --target libNeonPy -j 30