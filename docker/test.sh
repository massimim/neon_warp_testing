set -x
set -e

SCRIPT_HOME=${PWD}
git clone https://github.com/massimim/neon_warp_testing.git -b dev
cd ${SCRIPT_HOME}/neon_warp_testing
./init_https.sh
# building warp
cd ${SCRIPT_HOME}/neon_warp_testing/warp
export CUDA_PATH=/usr/local/cuda/
pip3 install --upgrade pip 
pip3 install numpy
python3 build_lib.py
pip3 install -e .
# Run tests
cd ${SCRIPT_HOME}/neon_warp_testing/testing
python3 test_00_index3d.py
python3 test_01_span.py
