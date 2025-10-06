set -e
set -x

if [ -z "$1" ]; then
  echo "Usage: $0 {build|compile} [debug|release]"
  echo "  build: Build the Neon Python bindings from scratch."
  echo "  compile: Compile the current build of the Neon Python bindings."
  echo "  [debug|release]: Optional. The build type. Default is 'release'."
  exit 1
fi

if [ "$1" != "build" ] && [ "$1" != "compile" ]; then
  echo "Error: Invalid argument. Use 'build' or 'compile'."
  exit 1
fi

BUILD_TYPE=${2:-release}

if [ "$BUILD_TYPE" != "debug" ] && [ "$BUILD_TYPE" != "release" ]; then
  echo "Error: Invalid build type. Use 'debug' or 'release'."
  exit 1
fi

if [ "$1" == "build" ]; then
  pushd .
  rm -fr ./neon/build
  mkdir -p ./neon/build
  cd ./neon/build
  # cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE^} -DNEON_ACTIVATE_TRACING=ON ..
  cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE^} ..
  popd
fi

pushd .
cd neon/build
cmake --build . --target libNeonPy -j 30
popd
