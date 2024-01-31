import argparse
import test_utils
import subprocess
import signal
from pathlib import Path

INT_MODEL_INPUTS = {
    "text/machine_comprehension/bert-squad/model/bertsquad-12.onnx": "--fill1 unique_ids_raw_output___9:0 --fill1 segment_ids:0 --fill1 input_mask:0 --fill1 input_ids:0".split(
        " "
    ),
    "text/machine_comprehension/bert-squad/model/bertsquad-10.onnx": "--fill1 unique_ids_raw_output___9:0 --fill1 segment_ids:0 --fill1 input_mask:0 --fill1 input_ids:0".split(
        " "
    ),
    "text/machine_comprehension/roberta/model/roberta-sequence-classification-9.onnx": "--fill1 input".split(
        " "
    ),
    "text/machine_comprehension/gpt-2/model/gpt2-10.onnx": "--fill1 input1 --input-dim @input1 1 1 8".split(
        " "
    ),
    "text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx": "--fill1 input1 --input-dim @input1 1 1 8".split(
        " "
    ),
    "text/machine_comprehension/t5/model/t5-encoder-12.onnx": "-fill1 input_ids --input-dim @input_ids 8 8".split(
        " "
    ),
    "text/machine_comprehension/t5/model/t5-decoder-with-lm-head-12.onnx": "-fill1 input_ids --input-dim @input_ids 8 8".split(
        " "
    ),
}


def get_models_with_accuracy_issue(onnx_zoo_path, file):
    models = {}
    with open(file, "r") as f:
        for line in f.readlines():
            if "Max diff(s):" in line:
                split_line = line.split("|")
                path = (
                    split_line[1]
                    .split("(")[1]
                    .split(")")[0]
                    .replace("tar.gz", "onnx")
                    .replace(
                        "https://github.com/gyulaz-htec/models/tree/migraphx_testing",
                        "",
                    )
                )
                max_diffs = split_line[5].strip()
                models[f"{onnx_zoo_path}{path}"] = max_diffs
    return models


def pull_models(models):
    test_utils.run_lfs_install()
    for model in models:
        test_utils.pull_lfs_file(model)


def process_verify_log(output):
    result = []
    relevant_stat = ["FAILED", "RMS Error", "Max diff", "Mismatch at", "Shape mismatch"]
    passed_message = "MIGraphX verification passed successfully"
    passed = False
    for line in output:
        line = line.decode("UTF-8")
        for stat in relevant_stat:
            if stat in line:
                result.append(line)

        if passed_message in line and len(result) == 0:
            passed = True


    if passed:
        return ["PASS"]
    else:
        return result


def verify_models(mgx_path, fp16, models):
    cmds = []
    for model in models:
        model = model.strip("/")
        cmd = [mgx_path, "verify", model]
        if model in INT_MODEL_INPUTS:
            cmd = cmd + INT_MODEL_INPUTS[model]
        if fp16:
            # skipping arcfaceresnet100-8 because it hangs
            if "arcfaceresnet100-8" in model:
                continue
            cmd.append("--fp16")
        cmds.append(cmd)

    for cmd in cmds:
        model = cmd[2]
        print(f"Executing: {' '.join(cmd)}")
        verification = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdoutput, stderror = verification.communicate()
        if verification.returncode == -signal.SIGSEGV:
            yield (model, ["Segmentation fault"])
        elif stderror:
            yield (model, ["Error"])
        else:
            yield (model, process_verify_log(stdoutput.splitlines()))


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

    parser.add_argument(
        "--output",
        required=False,
        default="result.txt",
        type=str,
        help="Name of the ouput file.",
    )

    args = parser.parse_args()
    models = get_models_with_accuracy_issue(
        args.onnx_zoo_path, "MIGRAPHX_FP16.md" if args.fp16 else "MIGRAPHX.md"
    )
    model_names = list(models.keys())
    pull_models(model_names)
    for model, stats in verify_models(args.mgx_path, args.fp16, model_names):
        with open(args.output, "+a") as o:
            o.write(f"\n{model}:")
            o.write(f"\n\tONNX Zoo:\n\t\t{models[f'/{model}']}")
            o.write(f"\n\tMIGraphX verify:")
            for stat in stats:
                o.write(f"\n\t\t{stat}")


if __name__ == "__main__":
    main()
