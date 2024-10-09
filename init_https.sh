set -e
set -x

rm -fr neon_py_bindings

git clone https://github.com/Autodesk/Neon.git -b py neon_py_bindings
git clone https://github.com/massimim/warp -b external-source-support warp_src
git clone https://github.com/massimim/XLB.git -b XLB_src

cd neon_py_bindings
mkdir cmake-build-debug
cd cmake-build-debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --target libNeonPy -j 30
