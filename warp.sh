set -e 
set -x
if [ -z "$1" ]; then
  echo "Usage: $0 {build|install}"
  echo "  build: Build the Warp project."
  echo "  install: Install the Warp project."
  exit 1
fi

if [ "$1" != "build" ] && [ "$1" != "install" ]; then
  echo "Error: Invalid argument. Use 'build' or 'install'."
  exit 1
fi

pushd .
cd warp

if [ "$1" == "build" ]; then
    echo "Building the Warp project..."
    echo "Using CUDA_PATH which is set to ${CUDA_PATH}"
    # Add build commands here
 
    pip3 install --upgrade pip 
    pip3 install numpy
    python3 build_lib.py
    pip3 install -e .
elif [ "$1" == "install" ]; then
    echo "Installing the Warp project..."
    # Add install commands here
    pip3 install -e .
fi

popd
