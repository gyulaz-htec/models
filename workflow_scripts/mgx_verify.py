import argparse
import test_utils
import subprocess
import signal
from pathlib import Path


def get_models_with_accuracy_issue(onnx_zoo_path, file):
    models = []
    with open(file, "r") as f:
        for line in f.readlines():
            if "Max diff(s):" in line:
                path = (
                    line.split("|")[1]
                    .split("(")[1]
                    .split(")")[0]
                    .replace("tar.gz", "onnx")
                    .replace(
                        "https://github.com/gyulaz-htec/models/tree/migraphx_testing",
                        "",
                    )
                )
                models.append(f"{onnx_zoo_path}{path}")
    return models


def pull_models(models):
    test_utils.run_lfs_install()
    for model in models:
        test_utils.pull_lfs_file(model)


def process_verify_log(output):
    result = []
    relevant_stat = ["FAILED", "RMS Error", "Max diff", "Mismatch at"]
    for line in output:
        line = line.decode("UTF-8")
        for stat in relevant_stat:
            if stat in line:
                result.append(line)
    return result


def verify_models(mgx_path, fp16, models):
    cmds = []
    statistics = {}
    for model in models:
        model = model.strip("/")
        cmd = [mgx_path, "verify", model]
        if fp16:
            # skipping arcfaceresnet100-8 because it hangs
            if "arcfaceresnet100-8" in model:
                continue
            cmd.append("--fp16")
        cmds.append(cmd)

    for cmd in cmds:
        model = cmd[-1]
        print(f"Executing: {' '.join(cmd)}")
        verification = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdoutput, stderror = verification.communicate()
        if verification.returncode == -signal.SIGSEGV:
            statistics[model] = ["Segmentation fault"]
        elif stderror:
            statistics[model] = ["Error"]
        else:
            statistics[model] = process_verify_log(stdoutput.splitlines())

    for model, stats in statistics.items():
        print(f"{model}:")
        for stat in stats:
            print(f"\t{stat}")


def main():
    parser = argparse.ArgumentParser(description="Verify settings")
    parser.add_argument(
        "--onnx-zoo-path",
        required=False,
        default="",
        type=str,
        help="The directory of ONXX Zoo models.",
    )

    parser.add_argument(
        "--mgx-path",
        required=True,
        default="",
        type=str,
        help="Path to migraphx-driver binary.",
    )

    parser.add_argument(
        "--fp16",
        required=False,
        default=False,
        action="store_true",
        help="Check fp16 results.",
    )

    args = parser.parse_args()
    models = get_models_with_accuracy_issue(
        args.onnx_zoo_path, "MIGRAPHX_FP16.md" if args.fp16 else "MIGRAPHX.md"
    )
    pull_models(models)
    verify_models(args.mgx_path, args.fp16, models)


if __name__ == "__main__":
    main()
