import glob
import os
import shutil
import mgx_inference as mgx
import numpy as np
import onnx
from onnx import numpy_helper
import json
import onnx_test_data_utils
from mgx_stats import MGXRunStats

def read_model_data_from_manifest(tar_gz_path):
    manifest_data = []
    with open('ONNX_HUB_MANIFEST.json') as f:
        manifest_data = json.load(f)
    model_data = next(filter(lambda model: model['metadata']['model_with_data_path'] == tar_gz_path, manifest_data), None)
    return model_data

def get_output_names_from_manifest(tar_gz_path):
    model_data = read_model_data_from_manifest(tar_gz_path)
    if not model_data:
        raise ValueError(f"No model data found in the manifest for archive: {tar_gz_path}")
    outputs = model_data['metadata']["io_ports"]["outputs"]
    output_names = [o["name"] for o in outputs]
    return output_names

def get_input_names_from_manifest(tar_gz_path,):
    model_data = read_model_data_from_manifest(tar_gz_path)
    if not model_data:
        raise ValueError(f"No model data found in the manifest for archive: {tar_gz_path}")
    inputs = model_data["metadata"]["io_ports"]["inputs"]
    input_names = [o["name"] for o in inputs]
    return input_names


def read_test_dir(dir_name, input_types, output_types):
    """
    Read the input and output .pb files from the provided directory.
    Input files should have a prefix of 'input_'
    Output files, which are optional, should have a prefix of 'output_'
    :param dir_name: Directory to read files from
    :return: tuple(dictionary of input name to numpy.ndarray of data,
                   dictionary of output name to numpy.ndarray)
    """

    inputs = {}
    outputs = {}

    input_files = glob.glob(os.path.join(dir_name, "input_*.pb"))
    output_files = glob.glob(os.path.join(dir_name, "output_*.pb"))

    if not len(input_files) == len(input_types):
        raise ValueError(f"Number of input files ({len(input_files)}) != number of model inputs ({len(input_types)}).")
    for i, filename in enumerate(input_files):
        if 'seq' in input_types[i]:
            name, data = onnx_test_data_utils.read_sequenceproto_pb_file(filename)
        else:
            name, data = onnx_test_data_utils.read_tensorproto_pb_file(filename)
        inputs[name] = data

    if not len(output_files) == len(output_types):
        raise ValueError(f"Number of output files ({len(output_files)}) != number of model outputs ({len(output_types)}).")
    for i, filename in enumerate(output_files):
        if 'seq' in output_types[i]:
            name, data = onnx_test_data_utils.read_sequenceproto_pb_file(filename)
        else:
            name, data = onnx_test_data_utils.read_tensorproto_pb_file(filename)
        outputs[name] = data

    return inputs, outputs

def save_outputs(tar_gz_path, expected, actual, output_name):
    file_base = tar_gz_path.replace(".tar.gz", "")
    expected_file = f"{file_base}_{output_name}_expected.txt"
    actual_file = f"{file_base}_{output_name}_actual.txt"
    np.savetxt(expected_file, expected)
    np.savetxt(actual_file, actual)
    print("Results written to:\n\tExpected:\t{}\n\tActual:\t\t{}".format(expected_file, actual_file))

def run_test_dir(model_or_dir, tar_gz_path, fp16, save_results):
    """
    Run the test/s from a directory in ONNX test format.
    All subdirectories with a prefix of 'test' are considered test input for one test run.

    :param model_or_dir: Path to onnx model in test directory,
                         or the test directory name if the directory only contains one .onnx model.

    :fp16: Wether to run with fp16 quantization or nit
    :tar_gz_path: Path to the .tar.gz file with the .onnx model and the example inputs and outputs
    :return: None
    """

    if os.path.isdir(model_or_dir):
        model_dir = os.path.abspath(model_or_dir)
        # check there's only one onnx file
        models = glob.glob(os.path.join(model_dir, "*.onnx"))
        if len(models) > 1:
            raise ValueError(
                "'Multiple .onnx files found in {}. '"
                "'Please provide specific .onnx file as input.".format(model_dir)
            )
        elif len(models) == 0:
            raise ValueError("'No .onnx file found in {}.".format(model_dir))

        model_path = models[0]
    else:
        model_path = os.path.abspath(model_or_dir)
        model_dir = os.path.dirname(model_path)

    stats = MGXRunStats(model_path)
    print("Running tests in {} for {}".format(model_dir, model_path))

    test_dirs = [d for d in glob.glob(os.path.join(model_dir, "test*")) if os.path.isdir(d)]
    if not test_dirs:
        raise ValueError("No directories with name starting with 'test' were found in {}.".format(model_dir))

    output_names_from_manifest = get_output_names_from_manifest(tar_gz_path)
    input_names_from_manifest = get_input_names_from_manifest(tar_gz_path)
    try:
        sess = mgx.session(model_path, fp16)
    except Exception as e:
        print(f"Fail during compilation: {str(e)}")
        stats.set_not_compiles(str(e))
        return stats

    input_types = [str(inp.type()) for inp in sess.get_inputs()]
    output_types = [str(out.type()) for out in sess.get_outputs()]

    for d in test_dirs:
        print(d)
        inputs, expected_outputs = read_test_dir(d, input_types, output_types)

        if set(inputs.keys()) != set(input_names_from_manifest):
            input_names = list(inputs.keys())
            input_names.sort()
            print(f"WARNING: Input names {input_names} are different from expected {input_names_from_manifest}. Trying to adjust them.")
            for i, name in enumerate(input_names):
                print(f"\tRenaming {name} to {input_names_from_manifest[i]}.")
                inputs[input_names_from_manifest[i]] = inputs[name]
                inputs.pop(name)

        if expected_outputs:
            if set(expected_outputs.keys()) != set(output_names_from_manifest):
                output_names = list(expected_outputs.keys())
                output_names.sort()
                print(f"WARNING: Output names {output_names} are different from expected {output_names_from_manifest}. Trying to adjust them.")
                for i, name in enumerate(output_names):
                    print(f"\tRenaming {name} ({expected_outputs[name].shape}) to {output_names_from_manifest[i]}.")
                    expected_outputs[output_names_from_manifest[i]] = expected_outputs[name]
                    expected_outputs.pop(name)

        try:
            run_outputs = sess.run(inputs)
        except Exception as e:
            stats.set_invalid(str(e))
            return stats

        if expected_outputs:
            print("Inference done. Validating outputs.")
            mismatch = False
            mismatch_message = "Max diff(s):"
            print('## 1')
            for idx in range(len(output_names_from_manifest)):
                output_name = output_names_from_manifest[idx]
                expected = expected_outputs[output_name]
                actual = np.array(run_outputs[idx])
                expected_shape = list(expected.shape)
                actual_shape = list(actual.shape)
                print(f'## 2 {output_name}')
                if (expected_shape != actual_shape):
                    print(f"WARNING: Output dimension differs from expected. Result:{actual_shape}, expected: {expected_shape}. Clipping result.")
                    same_dims = len(expected_shape) == len(actual_shape)
                    can_compare = same_dims
                    if same_dims:
                        for axis in range(len(expected_shape)):
                            if expected_shape[axis] > actual_shape[axis]:
                                can_compare = False
                                break
                            else:
                                actual = np.array(np.split(ary=actual, indices_or_sections=[expected_shape[axis],], axis=axis)[0])

                    if not can_compare:
                        print(f"WARNING: Clipping result failed, saving outputs.")
                        save_outputs(tar_gz_path, expected, actual, output_name)
                        stats.set_invalid(f"Output shape {actual_shape} doesn't match the expected {expected_shape}")
                        return stats

                print('## 3')
                if not np.isclose(expected, actual, rtol=1.0e-2, atol=1.0e-2).all():
                    mismatch = True
                    expected = expected.flatten('K')
                    actual = np.array(actual).flatten('K')
                    diff = expected - actual
                    max_diff = np.max(np.abs(diff))
                    mismatch_message = f"{mismatch_message} '{output_name}': {max_diff},"
                    print(f"Mismatch for {tar_gz_path}, output:{output_name}, max diff: {max_diff}")
                    if save_results:
                        save_outputs(tar_gz_path, expected, actual, output_name)

                print('## 4')
            if mismatch:
                stats.set_invalid(mismatch_message)

        return stats
