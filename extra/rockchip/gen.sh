#!/bin/bash -e
echo "building"
gcc -shared -fPIC -o preload_python.so preload.c -L/root/.pyenv/versions/3.11.4/lib -lpython3.11 -I/root/.pyenv/versions/3.11.4/include/python3.11
echo "compiled"
export LD_LIBRARY_PATH="/root/.pyenv/versions/3.11.4/lib"
export LD_PRELOAD="/root/tinygrad/extra/rockchip/preload_python.so"
export PYTHONPATH="/root/tinygrad"
# cd /data/snpe
# #ADSP_LIBRARY_PATH="." strace -f -e ioctl ./snpe-net-run --container MobileNetV2.dlc --input_list hello --use_dsp
# ADSP_LIBRARY_PATH="." ./snpe-net-run --container MobileNetV2.dlc --input_list hello --use_dsp

