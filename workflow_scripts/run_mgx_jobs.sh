#!/bin/sh

pip3 install py-cpuinfo onnxruntime py_markdown_table
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
apt-get install git-lfs

python3 -u workflow_scripts/test_models.py --all_models --target migraphx
python3 -u workflow_scripts/mgx_verify.py --mgx-path /code/AMDMIGraphX/build/bin/migraphx-driver --input MIGRAPHX_fp32.md --output fp32_verify.txt
python3 -u workflow_scripts/mgx_verify.py --mgx-path /code/AMDMIGraphX/build/bin/migraphx-driver --input MIGRAPHX_fp32-int8.md --output fp32_int8_verify.txt
python3 -u workflow_scripts/mgx_verify.py --mgx-path /code/AMDMIGraphX/build/bin/migraphx-driver --input MIGRAPHX_fp32-qdq.md --output fp32_qdq_verify.txt
python3 -u workflow_scripts/mgx_verify.py --mgx-path /code/AMDMIGraphX/build/bin/migraphx-driver --fp16 --input MIGRAPHX_fp16.md --output fp16_verify.txt
