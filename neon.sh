set -e 
set -x
if [ -z "$1" ]; then
  echo "Usage: $0 {build|compile}"
  echo "  build: Build the Neon Python bindings from scratch."
  echo "  compile: Compile the current builf of the Neon Python bindings."
  exit 1
fi

if [ "$1" != "build" ] && [ "$1" != "compile" ]; then
  echo "Error: Invalid argument. Use 'build' or 'compile'."
  exit 1
fi


if [ "$1" == "build" ]; then
  pushd .
  rm -fr ./neon/cmake-build-debug
  mkdir -p ./neon/cmake-build-debug
  cd ./neon/cmake-build-debug
  cmake -DCMAKE_BUILD_TYPE=Debug ..
  popd
fi

pushd .
cd neon/cmake-build-debug
cmake --build . --target libNeonPy -j 30
popd