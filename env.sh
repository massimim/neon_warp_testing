# set -e
# set -x

if [ -z "$1" ]; then
  echo "Usage: $0 {export|echo}"
  echo "  export: Export the environment variables."
  echo "  echo: Print the export operation."
  exit 1
fi

if [ "$1" != "export" ] && [ "$1" != "echo" ]; then
  echo "Error: Invalid argument. Use 'export' or 'echo'."
  exit 1
fi

PWD=`pwd`
PYTHON_SIDE="PYTHONPATH=$PWD/neon/py/:$PWD/XLB/:${PYTHON_PATH}"
CPP_SIDE="LD_LIBRARY_PATH=$PWD/neon/cmake-build-debug/lib/:${LD_LIBRARY_PATH}"

if [ "$1" == "export" ]; then
    export $PYTHON_SIDE
    export $CPP_SIDE
    echo "Variables exported."
elif [ "$1" == "echo" ]; then
    echo "export ${PYTHON_SIDE}"
    echo "export ${CPP_SIDE}"
fi

# set +e
# set +x