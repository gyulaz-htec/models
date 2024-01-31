#!/bin/sh

pip3 install py-cpuinfo onnxruntime py_markdown_table
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
apt-get install git-lfs

python3 -u workflow_scripts/test_models.py --all_models --target migraphx
python3 -u workflow_scripts/test_models.py --all_models --target migraphx --fp16
python3 -u workflow_scripts/mgx_verify.py --mgx-path /code/AMDMIGraphX/build/bin/migraphx-driver --output fp32_verify.txt
python3 -u workflow_scripts/mgx_verify.py --mgx-path /code/AMDMIGraphX/build/bin/migraphx-driver --fp16 --output fp16_verify.txt
