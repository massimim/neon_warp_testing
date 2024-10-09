cd warp_neon
export CUDA_PATH=/usr/local/cuda/
pip3 install --upgrade pip 
pip3 install numpy
python3 build_lib.py
pip3 install -e .
