import argparse
import os
import subprocess
from pathlib import Path

def create_command(line=str):
    cmd = "/code/AMDMIGraphX/build/bin/migraphx-driver verify vision/classification/shufflenet/model/shufflenet-v2-12.onnx --trim".split(' ')
    cmd.append(line)
    return cmd

def exec_command(cmd):
    verification = subprocess.Popen(
        cmd,
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdoutput, stderror = verification.communicate()
    return stdoutput.splitlines()

def output_has_fail(lines):
    for line in lines:
        try:
            line = line.decode()
        except:
            line = line.decode('latin1')
        if "FAILED:" in line:
            return True
    return False


def binary_search(good, bad):
    high = good
    low = bad
    mid = 0

    while low <= high:

        mid = (high + low) // 2
        cmd = create_command(str(mid))
        output = exec_command(cmd)
        fail = output_has_fail(output)
        print(f"Checked {mid}, Fails: {fail}")

        if fail:
            low = mid + 1
        else:
            high = mid - 1

    print(f"Failing line is at trim: {low - 1}")
    cmd = create_command(str(low - 1))
    print(f"MIGRAPHX_TRACE_EVAL=2 {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Test settings")
    # default all: test by both onnx and onnxruntime
    # if target is specified, only test by the specified one
    parser.add_argument(
        "--good",
        required=True,
        type=int,
        help="Test line in the model which had good outputs",
    )
    parser.add_argument(
        "--bad",
        required=True,
        type=int,
        help="Test line in the model which had bad outputs",
    )
    args = parser.parse_args()
    os.environ["MIGRAPHX_TRACE_EVAL"] = "2"

    binary_search(args.good, args.bad)

if __name__ == "__main__":
    main()
