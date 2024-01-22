# SPDX-License-Identifier: Apache-2.0

import argparse
import check_model
from pathlib import Path
import subprocess
import sys
import test_utils
import os
import markdown_utils
import mgx_stats



tar_ext_name = ".tar.gz"
onnx_ext_name = ".onnx"


def get_all_models():
    model_list = []
    for directory in ["text", "vision"]:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tar_ext_name) or file.endswith(onnx_ext_name):
                    model_list.append(os.path.join(root, file))
    return model_list


def get_changed_models():
    model_list = []
    cwd_path = Path.cwd()
    # git fetch first for git diff on GitHub Action
    subprocess.run(["git", "fetch", "origin", "main:main"],
                   cwd=cwd_path, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)
    # obtain list of added or modified files in this PR
    obtain_diff = subprocess.Popen(["git", "diff", "--name-only", "--diff-filter=AM", "origin/main", "HEAD"],
                                   cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdoutput, _ = obtain_diff.communicate()
    diff_list = stdoutput.split()

    # identify list of changed ONNX models in ONXX Model Zoo
    model_list = [str(model).replace("b'", "").replace("'", "")
                  for model in diff_list if onnx_ext_name in str(model) or tar_ext_name in str(model)]
    return model_list

def clean_up():
    test_utils.remove_onnxruntime_test_dir()
    test_utils.remove_tar_dir()
    test_utils.run_lfs_prune()

def check_migraphx_skip(model_name, fp16, opset):
    if "bertsquad-8.tar.gz" in model_name:
        print("Skipping model because it has mislabeled input/output proto buffer files.")
        return True
    if fp16:
        if ("MaskRCNN" in model_name or "FasterRCNN" in model_name):
            print("Skipping model because MaskRCNN and FasterRCNN models are crashing with --fp16.")
            return True
        if "int8" in model_name or "qdq" in model_name:
            print("Skipping model for --fp16 because it's already quantized.")
            return True
    if opset < 7:
        print("Skipping model because it has opset older than 7.")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Test settings")
    # default all: test by both onnx and onnxruntime
    # if target is specified, only test by the specified one
    parser.add_argument("--target", required=False, default="all", type=str,
                        help="Test the model by which (onnx/onnxruntime)?",
                        choices=["onnx", "onnxruntime", "migraphx", "all"])
    # use python workflow_scripts\test_models.py --create --all_models to create broken test data by ORT
    parser.add_argument("--create", required=False, default=False, action="store_true",
                        help="Create new test data by ORT if it fails with existing test data")
    parser.add_argument("--all_models", required=False, default=False, action="store_true",
                        help="Test all ONNX Model Zoo models instead of only changed models")
    parser.add_argument("--drop", required=False, default=False, action="store_true",
                        help="Drop downloaded models after verification. (For space limitation in CIs)")
    parser.add_argument("--fp16", required=False, default=False, action="store_true",
                        help="Enable fp16 quantization for migraphx")
    parser.add_argument("--model", required=False, default="", type=str,
                        help="Specify a target .tar.gz to run. Also enables save option.")
    parser.add_argument("--save", required=False, default=False, action="store_true",
                        help="Save expected and actual outputs for failing models")
    args = parser.parse_args()

    model_list = [args.model] if args.model else get_all_models() if args.all_models else get_changed_models()
    save = True if args.model else args.save
    # run lfs install before starting the tests
    test_utils.run_lfs_install()

    clean_up()

    print("\n=== Running test on ONNX models ===\n")
    failed_models = []
    skipped_models = []
    statistics = {}
    for model_path in model_list:
        model_name = model_path.split("/")[-1]
        print("==============Testing {}==============".format(model_name))

        try:
            # check .tar.gz by ORT and ONNX
            if tar_ext_name in model_name:
                # Step 1: check the ONNX model and test_data_set from .tar.gz by ORT
                test_data_set = []
                test_utils.pull_lfs_file(model_path)
                # check whether "test_data_set_0" exists
                model_path_from_tar, test_data_set = test_utils.extract_test_data(model_path)
                # if tar.gz exists, git pull and try to get test data
                if (args.target == "onnxruntime" or args.target == "all"):
                    # finally check the ONNX model from .tar.gz by ORT
                    # if the test_data_set does not exist, create the test_data_set
                    try:
                        check_model.run_backend_ort(model_path_from_tar, test_data_set)
                        print("[PASS] {} is checked by onnxruntime. ".format(model_name))
                    except Exception as e:
                        if not args.create:
                            raise
                        else:
                            print("Warning: original test data for {} is broken: {}".format(model_path, e))
                            test_utils.remove_onnxruntime_test_dir()
                        if (not model_name.endswith("-int8.tar.gz") and not model_name.endswith("-qdq.tar.gz")) or check_model.has_vnni_support():
                            check_model.run_backend_ort(model_path_from_tar, None, model_path)
                        else:
                            print("Skip quantized  models because their test_data_set was created in avx512vnni machines. ")
                        print("[PASS] {} is checked by onnxruntime. ".format(model_name))
                # Step 2: check the ONNX model inside .tar.gz by ONNX
                if args.target == "onnx" or args.target == "all":
                    check_model.run_onnx_checker(model_path_from_tar)
                    print("[PASS] {} is checked by onnx. ".format(model_name))
                # Step 3 check models with migraphx backend
                if args.target == "migraphx" or args.target == "all":
                    opset = mgx_stats.get_opset(model_path_from_tar)
                    skip = check_migraphx_skip(model_name, args.fp16, opset)
                    if skip:
                        skipped_models.append(model_path)
                        clean_up()
                        continue
                    try:
                        stats = check_model.run_backend_mgx(model_path_from_tar, test_data_set, model_path, args.fp16, save)
                        statistics[model_path] = stats
                    except Exception as e:
                        print("[FAIL] {}: {}".format(model_name, e))
                        failed_models.append(model_path)
                        statistics[model_path] = mgx_stats.MGXRunStats(model_path_from_tar, False, False, f"Error during script execution: {e}")
                        clean_up()
                        continue

                    if stats.valid:
                        print(f"[PASS] {model_name} is checked by migraphx.")
                    else:
                        print(f"[FAIL] {model_name}: {stats.error}")
                        failed_models.append(model_path)
            # check uploaded standalone ONNX model by ONNX
            elif onnx_ext_name in model_name:
                if args.target == "onnx" or args.target == "all":
                    test_utils.pull_lfs_file(model_path)
                    check_model.run_onnx_checker(model_path)
                    print("[PASS] {} is checked by onnx. ".format(model_name))

        except Exception as e:
            print("[FAIL] {}: {}".format(model_name, e))
            failed_models.append(model_path)

        # remove checked models and directories to save space in CIs
        if os.path.exists(model_path) and args.drop:
            os.remove(model_path)

        clean_up()

    markdown_utils.save_to_markdown(statistics, args.fp16)
    print("In all {} models, {} models failed and {} models were skipped. ".format(len(model_list), len(failed_models), len(skipped_models)))
    sys.exit(1)


if __name__ == "__main__":
    main()
